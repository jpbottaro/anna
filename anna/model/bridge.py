"""Bridges transform the state from the encoders so they fit the decoder

Inspired by OpenNMT & google/seq2seq

## Available bridges

@@NoBridge
@@ZeroBridge
@@DenseBridge
"""
import tensorflow as tf
import tensorflow.keras as tfk


class Bridge(tfk.layers.Layer):
    """Transforms the state from the encoders so they fit the decoder"""

    def call(self, inputs, **kwargs):
        """
        Call the underlying `bridge`

        Args:
            inputs ([tf.Tensor]): expects both zero_state and init_state
            **kwargs: none

        Returns:
            state (tf.Tensor): the value of the initial state
        """
        zero_state, init_state = inputs
        return self.bridge(zero_state, init_state)

    def bridge(self, zero_state, init_state):
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
    def bridge(self, zero_state, init_state):
        return init_state


class ZeroBridge(Bridge):
    def bridge(self, zero_state, init_state):
        return zero_state


class DenseBridge(Bridge):
    def build(self, input_shape):
        # Find sizes of all states
        flatten = tf.nest.flatten(input_shape[0])
        self.dims = [shape[-1] for shape in flatten]
        self.projection = tfk.layers.Dense(sum(self.dims))

    def bridge(self, zero_state, init_state):
        # Project `init` to cover all needed states
        states = self.projection(init_state)

        # Match dimensions of expected states
        states = tf.split(states, self.dims, axis=1)

        # Pack the result to conform with the requested states
        return tf.nest.pack_sequence_as(zero_state, states)
