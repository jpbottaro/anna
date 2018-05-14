"""Multi-label classification decoders using network-inspired architectures.

## Binary relevance decoder (independent binary classifier per label).

@@DecoderBR
"""
import tensorflow as tf


class Decoder:
    """
    Takes the output of an `Encoder`, and produces the final list
    of predictions.
    """

    def __call__(self, net, labels, mode):
        """
        Takes `net` as input and predicts the classes. Uses `labels` to
        generate a loss function to optimize the network.

        Args:
            net (tf.Tensor): the input to the decoder (coming from an `Encoder`)
            labels (tf.Tensor): the expected label output, as a 1/0 vector.
            mode (tf.estimator.ModeKeys): the mode we are on
        """
        raise NotImplementedError


class DecoderBR(Decoder):

    def __init__(self, data_dir, n_classes, hidden_units):
        """
        Binary Relevance decoder, where each label is an independent
        binary prediction.

        Args:
            data_dir (str): path to the data folder
            n_classes (int): number of classes to predict
            hidden_units (list[int]): size of each layer of the FFNN
        """
        self.n_classes = n_classes
        self.hidden_units = hidden_units

    def __call__(self, net, labels, mode):
        is_training = mode == tf.estimator.ModeKeys.TRAIN

        with tf.name_scope("decoder"):
            # Add all layers of the MLP
            for i, units in enumerate(self.hidden_units):
                net = tf.layers.dropout(net, training=is_training)
                net = tf.layers.dense(net, units=units, activation=tf.nn.relu)

            # Compute logits (1 per class)
            net = tf.layers.dropout(net, training=is_training)
            logits = tf.layers.dense(net, self.n_classes, activation=None)

            # Compute predictions as independent confidences
            probabilities = tf.nn.sigmoid(logits)

            # Threshold on 0.5
            predictions = tf.round(probabilities)

        loss = None
        if labels is not None:
            # Compute loss.
            with tf.name_scope("loss"):
                # Binary cross-entropy loss, with multiple labels per instance
                # (batch, n_classes)
                loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels,
                                                               logits=logits)

                # Average loss for all batches/classes
                # scalar
                loss = tf.reduce_mean(loss)

        return predictions, loss
