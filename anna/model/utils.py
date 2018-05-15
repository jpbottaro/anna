import tensorflow as tf


def rnn_cell(num_units, dropout, mode, residual=False, name=None, reuse=None):
    dropout = dropout if mode == tf.contrib.learn.ModeKeys.TRAIN else 0.0

    cell = tf.nn.rnn_cell.GRUCell(num_units, name=name, reuse=reuse)

    if dropout > 0.0:
        keep_prop = (1.0 - dropout)
        cell = tf.nn.rnn_cell.DropoutWrapper(
            cell=cell,
            input_keep_prob=keep_prop,
        )

    if residual:
        cell = tf.nn.rnn_cell.ResidualWrapper(cell)

    return cell
