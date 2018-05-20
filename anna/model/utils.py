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
    dropout = dropout if mode == tf.estimator.ModeKeys.TRAIN else 0.0

    if rnn_type == "lstm":
        cell = tf.nn.rnn_cell.LSTMCell(num_units, reuse=tf.AUTO_REUSE)
    elif rnn_type == "gru":
        cell = tf.nn.rnn_cell.GRUCell(num_units, reuse=tf.AUTO_REUSE)
    else:
        raise ValueError("Unknown rnn_type '{}'".format(rnn_type))

    if dropout > 0.0:
        keep_prop = (1.0 - dropout)
        cell = tf.nn.rnn_cell.DropoutWrapper(
            cell=cell,
            input_keep_prob=keep_prop,
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
