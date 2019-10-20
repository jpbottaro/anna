"""Multi-label classification decoders using network-inspired architectures.

## Binary relevance decoder (independent binary classifier per label).

@@DecoderBR

## Sequence prediction decoder (one label prediction per timestep).

@@DecoderRNN
@@DecoderAttRNN
"""
import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow_addons.seq2seq as s2s
import anna.model.utils as utils
from anna.model.bridge import DenseBridge


class Decoder(tfk.layers.Layer):
    """
    Takes the output of an `Encoder`, and produces the final list
    of predictions.
    """

    def __init__(self, name="decoder"):
        """
        Creates an decoder with the given name.

        Args:
            name (str): name of the decoder
        """
        super(Decoder, self).__init__(name=name)

    def get_config(self):
        """
        Gets the configuration of the decoder

        Returns:
            config (dict[str, str]): configuration of the decoder
        """
        return {
            'type': str(type(self))
        }

    def call(self, inputs, training=None):
        """
        Takes `net` as input and predicts the classes. Uses `labels` to
        generate a loss function to optimize the network.

        Args:
            inputs (list[tf.Tensor]): a list of tensors, expecting:
                - mem (tf.Tensor): sequential representation of the input.
                  [batch, sum(len), size]
                - mem_len (tf.Tensor): length of `mem`.
                  [batch]
                - mem_fixed (tf.Tensor): fixed-sized representation of the input.
                  [batch, size]
                - labels (tf.Tensor): the expected label output, as a 1/0 vector.
                  [batch, n_labels]
            training (bool): if this is training or eval

        Returns:
            predictions (tf.Tensor): multi-hot tensor of predictions
            loss (tf.Tensor): loss function for the network
        """
        mem, mem_len, mem_fixed, labels = inputs
        with tf.name_scope("decoder"):
            predictions, loss = self.decode(mem, mem_len, mem_fixed, labels, training)
            self.add_loss(loss)
            return predictions

    def decode(self, mem, mem_len, mem_fixed, labels, training):
        """
        Takes `net` as input and predicts the classes.

        Args:
            mem (tf.Tensor): sequential representation of the input.
              [batch, sum(len), size]
            mem_len (tf.Tensor): length of `mem`.
              [batch]
            mem_fixed (tf.Tensor): fixed-sized representation of the input.
              [batch, size]
            labels (tf.Tensor): the expected label output, as a 1/0 vector.
              [batch, n_labels]
            training (bool): if this is training or eval

        Returns:
            predictions (tf.Tensor): multi-hot tensor of predictions
            loss (tf.Tensor): loss function for the network
        """
        raise NotImplementedError


class DecoderBR(Decoder):

    def __init__(self, data_dir, n_classes, hidden_units, dropout=.5):
        """
        Binary Relevance decoder, where each label is an independent
        binary prediction.

        Args:
            data_dir (str): path to the data folder
            n_classes (int): number of classes to predict
            hidden_units (list[int]): size of each layer of the FFNN
            dropout (float): dropout rate for each layer
        """
        super(DecoderBR, self).__init__()
        _ = data_dir

        self.net = tfk.models.Sequential()
        for i, units in enumerate(hidden_units):
            self.net.add(tfk.layers.Dropout(rate=dropout))
            self.net.add(tfk.layers.Dense(units, activation=tf.nn.relu))
        self.net.add(tfk.layers.Dropout(rate=dropout))
        self.net.add(tfk.layers.Dense(n_classes, activation=None))

    def decode(self, mem, mem_len, mem_fixed, labels, training):
        # Compute logits for each class
        logits = self.net(mem_fixed)

        # Compute predictions as independent confidences
        probabilities = tf.nn.sigmoid(logits)

        # Threshold on 0.5
        predictions = tf.round(probabilities)

        # Compute loss
        loss = None
        if labels is not None:
            with tf.name_scope("loss"):
                # Binary cross-entropy loss, with multiple labels per instance
                # (batch, n_classes)
                loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels,
                                                               logits=logits)

                # Average loss for all batches/classes
                # scalar
                loss = tf.reduce_mean(input_tensor=loss)

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
                 dropout=.3,
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
            attention (tfa.seq2seq.AttentionMechanism): the attention
              mechanism to use (e.g. LuongAttention)
            loss (func): function that returns the loss for the model.
        """
        super(DecoderRNN, self).__init__()
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

    def decode(self, mem, mem_len, mem_fixed, labels, training):
        with tf.name_scope("decoder"):
            n_labels = len(self.voc)
            batch_size = tf.shape(input=mem)[0]

            target, target_len, target_max_len = self.encode_labels(labels)

            output_layer = tfk.layers.Dense(n_labels)
            cell, cell_init = self.build_cell(mem, mem_len, mem_fixed, training)
            emb = tf.Variable(name="label_embeddings",
                              shape=[n_labels, self.emb_size],
                              initial_value=tf.initializers.glorot_uniform)

            # Training
            if training:
                # Shift targets to the right, adding the start token
                # [batch, steps]
                start_tokens = tf.fill([batch_size, 1], self.sos_id)
                inputs = tf.concat([start_tokens, target[:, :-1]], axis=1)

                # [batch, steps, emb_size]
                inputs = tf.nn.embedding_lookup(params=emb, ids=inputs)

                # Decoder
                sampler = s2s.sampler.TrainingSampler()
                decoder = s2s.BasicDecoder(cell, sampler)

                outputs, _, _ = decoder(
                    inputs,
                    initial_state=cell_init,
                    sequence_length=target_len)

                logits = output_layer(outputs.rnn_output)

            # Inference
            else:
                start_tokens = tf.fill([batch_size], self.sos_id)

                if self.beam_width > 0:
                    dec = s2s.BeamSearchDecoder(
                        cell=cell,
                        beam_width=self.beam_width,
                        output_layer=output_layer
                    )
                else:
                    dec = s2s.BasicDecoder(
                        cell,
                        s2s.GreedyEmbeddingSampler(emb),
                        cell_init,
                        output_layer=output_layer,
                    )

                outputs, _ = dec(
                    emb,
                    start_tokens=start_tokens,
                    end_token=self.eos_id,
                    initial_state=cell_init,
                    training=training
                )

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
                if not training:
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
                loss = tf.reduce_sum(input_tensor=loss * mask)

                # normalize by batch size
                loss /= tf.cast(batch_size, tf.float32)

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
        labels = tf.cast(labels, tf.int32)
        mask, indices = tf.nn.top_k(labels, k=self.max_steps - 1)

        # Fix index indices to account for special labels
        # (batch, max_steps)
        indices = (indices + self.n_special) * mask

        # Length of each list of labels
        # (batch)
        seq_len = tf.reduce_sum(input_tensor=mask, axis=1)

        # Limit size of target to longest sequence
        # (batch, max_steps)
        max_len = tf.reduce_max(input_tensor=seq_len)
        indices = indices[:, :max_len]

        # Add extra column for EOS label
        # (batch, max_steps)
        indices = tf.pad(tensor=indices, paddings=[[0, 0], [0, 1]])

        # Add EOS label
        # (batch, max_steps)
        batch_size = tf.shape(input=labels)[0]
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
        predictions = tf.argmax(input=logits, axis=2)
        max_len = tf.shape(input=predictions)[1]

        # Build mask until the first EOS label
        # (batch_size, n_steps)
        mask = tf.where(tf.equal(predictions, self.eos_id),
                        tf.ones_like(predictions),
                        tf.zeros_like(predictions))

        # argmax will keep the first hit
        # (batch_size)
        eos_idx = tf.argmax(input=mask, axis=1)

        # Only allow predictions until first EOS
        # (batch_size, n_steps)
        mask = tf.sequence_mask(eos_idx,
                                max_len,
                                dtype=tf.int64)

        # (batch_size, n_steps)
        predictions = predictions * mask

        # Transform each label index into one-hot
        # (batch_size, n_steps, n_labels + n_special)
        predictions = tf.one_hot(predictions, len(self.voc), dtype=tf.float32)

        # Add all one-hot to build multi-hot vector
        # (batch_size, n_labels + n_special)
        predictions = tf.reduce_sum(input_tensor=predictions, axis=1)

        # If labels were predicted multiple times, just make it a 1
        # (batch_size, n_labels + n_special)
        predictions = tf.minimum(predictions, 1.)

        # Remove special labels from the predictions
        # (batch_size, n_labels)
        return predictions[:, self.n_special:]

    def build_cell(self, mem, mem_len, mem_fixed, training):
        """
        Builds the RNN cell that will drive the decoder.

        Args:
            mem (tf.Tensor): sequential representation of the input.
              [batch, sum(len), size]
            mem_len (tf.Tensor): length of `mem`.
              [batch]
            mem_fixed (tf.Tensor): fixed-sized representation of the input.
              [batch, size]
            training (bool): if this is training or eval

        Returns:
            cell (tf.nn.rnn_cell.RNNCell): the final RNN cell
        """
        cell = utils.rnn_cell(self.rnn_type,
                              self.hidden_size,
                              training,
                              self.dropout)

        # Build initial state based on `mem_fixed`
        batch_size = tf.shape(input=mem_fixed)[0]
        zero_state = cell.zero_state(batch_size, mem.dtype)
        init = self.bridge(zero_state, mem_fixed)

        if not training and self.beam_width > 0:
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

            if training and self.dropout > 0.:
                cell = tf.nn.RNNCellDropoutWrapper(
                    cell=cell,
                    output_keep_prob=1. - self.dropout
                )

            init = cell.zero_state(batch_size, mem.dtype)

        return cell, init
