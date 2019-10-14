"""Bridges transform the state from the encoders so they fit the decoder

Inspired by OpenNMT & google/seq2seq

## Available bridges

@@NoBridge
@@ZeroBridge
@@DenseBridge
"""
import tensorflow as tf
import tensorflow.keras as keras


class Bridge:
    """Transforms the state from the encoders so they fit the decoder"""

    def __call__(self, zero_state, init_state):
        """
        Creates a state for a cell that accepts `zero_state` type of states. Uses
        `init` as the input.

        Args:
            zero_state (tf.Tensor): the result of cell#zero_state().
            init_state (tf.Tensor): initialization for the state.
              [batch_size, size]

        Returns:
            init_state (tf.Tensor): same size as `zero_state`, initialized with
              `init_state`.
        """
        raise NotImplementedError


class NoBridge(Bridge):
    def __call__(self, zero_state, init_state):
        return init_state


class ZeroBridge(Bridge):
    def __call__(self, zero_state, init_state):
        return zero_state


class DenseBridge(Bridge):
    def __call__(self, zero_state, init_state):
        # See states as a flat list of tensors
        zero_state_flat = tf.nest.flatten(zero_state)

        # Find sizes of all states
        dims = [t.get_shape()[-1].value for t in zero_state_flat]

        # Project `init` to cover all needed states
        states = keras.layers.Dense(sum(dims))(init_state)

        # Match dimensions of expected states
        states = tf.split(states, dims, axis=1)

        # Pack the result to conform with the requested states
        return tf.nest.pack_sequence_as(zero_state, states)
