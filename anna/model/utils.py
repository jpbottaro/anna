import tensorflow as tf


def rnn_cell(rnn_type, num_units, mode, dropout=0., residual=False):
    """
    Creates an RNN cell of type `rnn_type`.

    Supports "lstm" and "gru".

    Args:
        rnn_type (str): the type of RNN ("lstm" or "gru")
        num_units (int): num of hidden units for the cell
        mode (tf.estimator.ModeKeys): the current mode
        dropout (float, optional): percentage to apply for dropout
        residual (bool, optional): whether to use residual connections

    Returns:
        cell (tf.nn.rnn_cell.RNNCell): an RNN cell
    """
    dropout = dropout if mode == tf.estimator.ModeKeys.TRAIN else 0.

    if rnn_type == "lstm":
        cell = tf.nn.rnn_cell.LSTMCell(num_units)
    elif rnn_type == "gru":
        cell = tf.nn.rnn_cell.GRUCell(num_units)
    else:
        raise ValueError("Unknown rnn_type '{}'".format(rnn_type))

    if dropout > 0.:
        keep_prob = (1. - dropout)
        cell = tf.nn.rnn_cell.DropoutWrapper(
            cell=cell,
            input_keep_prob=keep_prob,
        )

    if residual:
        cell = tf.nn.rnn_cell.ResidualWrapper(cell)

    return cell


def rnn_state_trim(rnn_type, states):
    """
    Trims a given state depending on the type of RNN.

    It mostly affects LSTMs, which return a tuple as a state, by only returning
    the hidden vector.

    Args:
        rnn_type (str): type of RNN (e.g. "lstm").
        states: the states returned by the RNN.
          [2, batch_size, size] for LSTM
          [batch_size, size] for others

    Returns:
        states: the trimmed states.
          [batch_size, size]
    """
    if rnn_type == "lstm":
        return states[0]

    return states


def seq_pad(x, size):
    """
    Pads a sequential tensor with zeros up to `size` steps.

    Args:
        x (tf.Tensor): the tensor we want to pad
          [batch, steps, emb_size]
        size (int): size to pad

    Returns:
        x (tf.Tensor): the padded tensor
          [batch, size, emb_size]
    """
    emb_size = x.get_shape()[-1].value
    x = tf.pad(x, [[0, 0], [0, size], [0, 0]])
    x = x[:, :size, :]
    x.set_shape([None, None, emb_size])

    return x


def seq_roll(x, size):
    """
    Rolls the input `x` by `size` positions to the right, wrapping around.

    Similar to OpenNMT's roll_sequence.

    Args:
        x (tf.Tensor): the tensor we want to roll
          [batch, steps, emb_size]
        size (tf.Tensor): how many positions we want to roll `x` to the right
          [batch]

    Returns:
        y (tf.Tensor): the rolled tensor
          [batch, steps, emb_size]
    """
    batch_size = tf.shape(x)[0]
    steps = tf.shape(x)[1]

    # Build grid with all indices in the tensor `x`
    # x = [1 3 3 7]
    # size = 2
    cols, rows = tf.meshgrid(tf.range(steps), tf.range(batch_size))

    # Subtract the amount we want to roll
    # [-2 -1 0 1]
    cols -= tf.expand_dims(size, 1)

    # Take the remainder of the division. This returns the rolled indices, e.g.
    # [2 3 0 1]
    cols = tf.floormod(cols, steps)

    # Build back the indices using the rolled targets
    indices = tf.stack([rows, cols], axis=-1)

    # Shuffle the input with the constructed indices
    return tf.gather_nd(x, indices)


def seq_concat(memory, memory_len):
    """
    Concatenates a list of sequential memories into a single sequence.

    Similar to OpenNMT's ConcatReducer.

    Args:
        memory (list[tf.Tensor]): a list of sequential tensors to concatenate.
          [batch, len, size]
        memory_len (list[tf.Tensor]): the length tensor in `memory`.
          [batch]

    Returns:
        final_mem (tf.Tensor): sequential representation of the input.
          [batch, sum(len), size]
        final_mem_len (tf.Tensor): length of `final_mem`.
          [batch]
    """
    if len(memory) == 1:
        return memory[0], memory_len[0]

    # Calculate final length of each instance in the batch
    # [batch]
    final_len = tf.add_n(memory_len)

    # Calculate largest instance
    max_len = tf.reduce_max(final_len)

    # Pad memory to the largest combined sequence
    # list([batch, max_len, emb_size])
    memory = [seq_pad(m, max_len) for m in memory]

    # Build padded final memory
    # [batch, max_len, emb_size]
    final_mem = None
    accum_len = None

    # Roll and accumulate sequences to obtain the final concatenation
    for x, x_len in zip(memory, memory_len):
        if final_mem is None:
            final_mem = x
            accum_len = x_len
        else:
            final_mem += seq_roll(x, accum_len)
            accum_len += x_len

    return final_mem, final_len
