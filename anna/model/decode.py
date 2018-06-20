"""Multi-label classification decoders using network-inspired architectures.

## Binary relevance decoder (independent binary classifier per label).

@@DecoderBR

## Sequence prediction decoder (one label prediction per timestep).

@@DecoderRNN
@@DecoderAttRNN
"""
import tensorflow as tf
import tensorflow.contrib.seq2seq as s2s
import anna.model.utils as utils
from anna.model.bridge import DenseBridge


class Decoder:
    """
    Takes the output of an `Encoder`, and produces the final list
    of predictions.
    """

    def __call__(self, mem, mem_len, mem_fixed, labels, mode):
        """
        Takes `net` as input and predicts the classes. Uses `labels` to
        generate a loss function to optimize the network.

        Args:
            mem (tf.Tensor): sequential representation of the input.
              [batch, sum(len), size]
            mem_len (tf.Tensor): length of `mem`.
              [batch]
            mem_fixed (tf.Tensor): fixed-sized representation of the input.
              [batch, size]
            labels (tf.Tensor): the expected label output, as a 1/0 vector.
              [batch, n_labels]
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
        _ = data_dir

        self.n_classes = n_classes
        self.hidden_units = hidden_units

    def __call__(self, mem, mem_len, mem_fixed, labels, mode):
        is_training = mode == tf.estimator.ModeKeys.TRAIN

        net = mem_fixed
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


class DecoderRNN(Decoder):
    """
    Decoder as a sequence prediction, where a new label is predicted at each
    time step (until a special end labels is produced).
    """

    def __init__(self,
                 data_dir,
                 label_voc,
                 hidden_size=1024,
                 max_steps=30,
                 emb_size=256,
                 rnn_type="gru",
                 bridge=DenseBridge(),
                 dropout=.2,
                 beam_width=0,
                 attention=None,
                 loss=tf.nn.sparse_softmax_cross_entropy_with_logits):
        """
        Creates the sequence prediction decoder.

        Args:
            data_dir (str): path to the data folder.
            label_voc (list[str]): vocabulary of classes to predict.
            hidden_size (int): hidden size of the decoding RNN.
            max_steps (int): max number of steps the decoder can take.
            emb_size (int): size of label embeddings.
            rnn_type (str): type of rnn (options: "gru" or "lstm").
            bridge (Bridge): how to hook the input to the RNN state.
            dropout (float): rate of dropout to apply (0 to disable).
            beam_width (int): size of the beam search beam (0 to disable).
            attention (tf.contrib.seq2seq.AttentionMechanism): the attention
              mechanism to use (e.g. LuongAttention)
            loss (func): function that returns the loss for the model.
        """
        _ = data_dir

        self.hidden_size = hidden_size
        self.max_steps = max_steps
        self.emb_size = emb_size
        self.rnn_type = rnn_type
        self.dropout = dropout
        self.bridge = bridge
        self.beam_width = beam_width
        self.attention = attention
        self.loss = loss

        special_labels = ["_PAD_", "_SOS_", "_EOS_"]
        self.voc = special_labels + label_voc
        self.sos_id = 1
        self.eos_id = 2
        self.n_special = len(special_labels)

    def __call__(self, mem, mem_len, mem_fixed, labels, mode):
        is_training = mode == tf.estimator.ModeKeys.TRAIN
        is_eval = mode == tf.estimator.ModeKeys.EVAL

        with tf.variable_scope("decoder") as scope:
            n_labels = len(self.voc)
            batch_size = tf.shape(labels)[0]

            target, target_len, target_max_len = self.encode_labels(labels)

            output_layer = tf.layers.Dense(n_labels)
            cell, cell_init = self.build_cell(mem, mem_len, mem_fixed, mode)
            emb = tf.get_variable("label_embeddings", [n_labels, self.emb_size])

            # Training
            if is_training:
                # Shift targets to the right, adding the start token
                # [batch, steps]
                start_tokens = tf.fill([batch_size, 1], self.sos_id)
                inputs = tf.concat([start_tokens, target[:, :-1]], axis=1)

                # [batch, steps, emb_size]
                inputs = tf.nn.embedding_lookup(emb, inputs)

                helper = s2s.TrainingHelper(inputs, target_len)

                dec = s2s.BasicDecoder(cell, helper, cell_init)

                outputs, _, _ = s2s.dynamic_decode(
                    dec,
                    maximum_iterations=self.max_steps,
                    swap_memory=True,
                    scope=scope)

                # [batch, steps, n_classes]
                logits = output_layer(outputs.rnn_output)

            # Inference
            else:
                start_tokens = tf.fill([batch_size], self.sos_id)

                if self.beam_width > 0:
                    dec = s2s.BeamSearchDecoder(
                        cell=cell,
                        embedding=emb,
                        start_tokens=start_tokens,
                        end_token=self.eos_id,
                        initial_state=cell_init,
                        beam_width=self.beam_width,
                        output_layer=output_layer)
                else:
                    helper = s2s.GreedyEmbeddingHelper(
                        emb, start_tokens, self.eos_id)

                    dec = s2s.BasicDecoder(
                        cell,
                        helper,
                        cell_init,
                        output_layer=output_layer
                    )

                outputs, _, _ = s2s.dynamic_decode(
                    dec,
                    maximum_iterations=self.max_steps,
                    swap_memory=True,
                    scope=scope)

                # [batch, steps, n_classes]
                if self.beam_width > 0:
                    logits = tf.one_hot(outputs.predicted_ids[:, :, 0],
                                        n_labels)
                else:
                    logits = outputs.rnn_output

        predictions = self.decode_labels(logits)

        loss = None
        if labels is not None:
            with tf.name_scope("loss"):
                # If this is eval, make the loss only look at the target steps
                if is_eval:
                    pads = tf.zeros([batch_size, self.max_steps, n_labels])
                    logits = tf.concat([logits, pads], axis=1)
                    logits = logits[:, :target_max_len, :]

                # [batch, steps, n_classes]
                mask = tf.sequence_mask(target_len,
                                        target_max_len,
                                        dtype=logits.dtype)

                # [batch, steps, n_classes]
                loss = self.loss(labels=target, logits=logits)

                # scalar
                loss = tf.reduce_sum(loss * mask) / tf.to_float(batch_size)

        return predictions, loss

    def encode_labels(self, labels):
        """
        Transforms the multi-hot expected labels into a list of ordered labels
        to use as targets for the sequential decoding.

        Args:
            labels (tf.Tensor): the expected labels, encoded as multi-hot
              (batch_size, n_labels).

        Returns:
            target (tf.Tensor): the targets, as list of positive labels
              (batch_size, n_steps).
            target_len (tf.Tensor): size of each list of positive labels
              (batch_size).
            max_len (tf.Tensor): maximum length of predictions.
        """
        # Since labels is multi-hot, `top_k` retrieves the hot indices, and
        # values act as a mask.
        # labels:  (batch, n_labels)      ~ [[0, 1, 1], [1, 0, 0]]
        # mask:    (batch, max_steps - 1) ~ [[1, 1, 0], [1, 0, 0]]
        # indices: (batch, max_steps - 1) ~ [[1, 2, 0], [0, 1, 2]]
        labels = tf.to_int32(labels)
        mask, indices = tf.nn.top_k(labels, k=self.max_steps - 1)

        # Fix index indices to account for special labels
        # (batch, max_steps)
        indices = (indices + self.n_special) * mask

        # Length of each list of labels
        # (batch)
        seq_len = tf.reduce_sum(mask, axis=1)

        # Limit size of target to longest sequence
        # (batch, max_steps)
        max_len = tf.reduce_max(seq_len)
        indices = indices[:, :max_len]

        # Add extra column for EOS label
        # (batch, max_steps)
        indices = tf.pad(indices, [[0, 0], [0, 1]])

        # Add EOS label
        # (batch, max_steps)
        batch_size = tf.shape(labels)[0]
        cols = tf.range(batch_size)
        eos_indices = tf.stack([cols, seq_len], axis=1)
        eos = tf.scatter_nd(indices=eos_indices,
                            updates=tf.fill([batch_size], self.eos_id),
                            shape=[batch_size, max_len + 1])

        return indices + eos, seq_len + 1, max_len + 1

    def decode_labels(self, logits):
        """
        Transforms the sequence of predictions into a multi-hot vector of
        labels.

        Args:
            logits (tf.Tensor): the sequence of logits for each RNN prediction
              (batch_size, n_steps, n_labels).

        Returns:
            predictions (tf.Tensor): multi-hot encoding of the label predictions
              (batch_size, n_labels).
        """
        # Take highest scoring labels per step
        # (batch_size, n_steps)
        predictions = tf.argmax(logits, 2)
        max_len = tf.shape(predictions)[1]

        # Build mask until the first EOS label
        # (batch_size, n_steps)
        mask = tf.where(tf.equal(predictions, self.eos_id),
                        tf.ones_like(predictions),
                        tf.zeros_like(predictions))

        # argmax will keep the first hit
        # (batch_size)
        mask = tf.argmax(mask, 1)

        # Only allow predictions until first EOS
        # (batch_size, n_steps)
        mask = tf.sequence_mask(mask,
                                max_len,
                                dtype=tf.int64)

        # (batch_size, n_steps)
        predictions = predictions * mask

        # Transform each label index into one-hot
        # (batch_size, n_steps, n_labels + n_special)
        predictions = tf.one_hot(predictions, len(self.voc), dtype=tf.float32)

        # Add all one-hot to build multi-hot vector
        # (batch_size, n_labels + n_special)
        predictions = tf.reduce_sum(predictions, 1)

        # If labels were predicted multiple times, just make it a 1
        # (batch_size, n_labels + n_special)
        predictions = tf.where(tf.greater(predictions, 0),
                               tf.ones_like(predictions),
                               tf.zeros_like(predictions))

        # Remove special labels from the predictions
        # (batch_size, n_labels)
        return predictions[:, self.n_special:]

    def build_cell(self, mem, mem_len, mem_fixed, mode):
        """
        Builds the RNN cell that will drive the decoder.

        Args:
            mem (tf.Tensor): sequential representation of the input.
              [batch, sum(len), size]
            mem_len (tf.Tensor): length of `mem`.
              [batch]
            mem_fixed (tf.Tensor): fixed-sized representation of the input.
              [batch, size]
            mode (tf.estimator.ModeKeys): the mode we are on

        Returns:
            cell (tf.nn.rnn_cell.RNNCell): the final RNN cell
        """
        is_training = mode == tf.estimator.ModeKeys.TRAIN
        cell = utils.rnn_cell(self.rnn_type,
                              self.hidden_size,
                              mode,
                              self.dropout)

        # Build initial state based on `mem_fixed`
        batch_size = tf.shape(mem_fixed)[0]
        zero_state = cell.zero_state(batch_size, mem.dtype)
        init = self.bridge(zero_state, mem_fixed)

        if not is_training and self.beam_width > 0:
            init = s2s.tile_batch(init, multiplier=self.beam_width)
            mem = s2s.tile_batch(mem, multiplier=self.beam_width)
            mem_len = s2s.tile_batch(mem_len, multiplier=self.beam_width)
            batch_size *= self.beam_width

        if self.attention:
            att_mechanism = self.attention(self.hidden_size, mem, mem_len)

            cell = s2s.AttentionWrapper(
                cell,
                att_mechanism,
                attention_layer_size=self.hidden_size,
                initial_cell_state=init,
                name="attention")

            init = cell.zero_state(batch_size, mem.dtype)

        return cell, init
