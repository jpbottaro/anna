import tensorflow as tf
import anna.model.utils as utils


class SequenceUtilsTest(tf.test.TestCase):
    def test_seq_roll(self):
        with self.test_session():
            x = tf.constant([[0, 1, 0], [1, 0, 1], [0, 0, 1]])
            size = tf.constant([1, 0, 3])
            y = utils.seq_roll(x, size)

            self.assertAllEqual(y.eval(), [[0, 0, 1], [1, 0, 1], [0, 0, 1]])

    def test_seq_roll(self):
        with self.test_session():
            x = tf.constant([[0, 1, 0], [1, 0, 1], [0, 0, 1]])
            x = tf.expand_dims(x, axis=-1)

            small = utils.seq_pad(x, 1)
            same = utils.seq_pad(x, 3)
            large = utils.seq_pad(x, 5)

            self.assertAllEqual(small.eval(), [[[0]], [[1]], [[0]]])
            self.assertAllEqual(same.eval(), x.eval())
            self.assertAllEqual(large.eval(), [[[0], [1], [0], [0], [0]],
                                               [[1], [0], [1], [0], [0]],
                                               [[0], [0], [1], [0], [0]]])

    def test_seq_concat(self):
        with self.test_session():
            x = tf.constant([[2, 1, 0], [1, 3, 1], [1, 0, 0]])
            x = tf.expand_dims(x, axis=-1)
            x_len = tf.constant([2, 3, 1])

            y = tf.constant([[1, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]])
            y = tf.expand_dims(y, axis=-1)
            y_len = tf.constant([2, 1, 0])

            res, res_len = utils.seq_concat([x, y], [x_len, y_len])

            self.assertAllEqual(res.eval(), [[[2], [1], [1], [1]],
                                             [[1], [3], [1], [1]],
                                             [[1], [0], [0], [0]]])
            self.assertAllEqual(res_len.eval(), [4, 4, 1])
