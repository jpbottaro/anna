import tensorflow as tf
from anna.model.decode import DecoderRNN


class EncoderDecoderTest(tf.test.TestCase):
    def test_encode(self):
        labels = ["one", "two", "three"]
        dec = DecoderRNN("no/value", labels, 128, max_steps=len(labels))

        x = tf.constant([[0, 1, 0], [1, 0, 1]])
        x, x_len, x_max = dec.encode_labels(x)

        self.assertAllEqual(x, [[4, 2, 0], [3, 5, 2]])
        self.assertAllEqual(x_len, [2, 3])
        self.assertAllEqual(x_max, 3)

    def test_decode(self):
        labels = ["one", "two", "three"]
        dec = DecoderRNN("no/value", labels, 128, max_steps=len(labels))

        x = tf.constant([[3, 2, 0], [2, 0, 1], [5, 4, 2]])
        x = tf.one_hot(x, len(dec.voc))
        x = dec.decode_labels(x)

        self.assertAllEqual(x, [[1, 0, 0], [0, 0, 0], [0, 1, 1]])

    def test_encode_decode(self):
        labels = ["one", "two", "three"]
        dec = DecoderRNN("no/value", labels, 128, max_steps=len(labels))

        orig_x = tf.constant([[0, 1, 0], [1, 0, 1]], dtype=tf.float32)
        x, x_len, x_max = dec.encode_labels(orig_x)
        x = tf.one_hot(x, len(dec.voc))
        x = dec.decode_labels(x)

        self.assertAllEqual(orig_x, x)

    def test_repeat_decode(self):
        labels = ["one", "two", "three"]
        dec = DecoderRNN("no/value", labels, 128, max_steps=len(labels))

        x = tf.constant([[3, 3, 2, 0]])
        x = tf.one_hot(x, len(dec.voc))
        x = dec.decode_labels(x)

        self.assertAllEqual(x, [[1, 0, 0]])
