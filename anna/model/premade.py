from anna.model.trainer import Trainer
from anna.model.encode import *
from anna.model.decode import *


class AVGxBR(Trainer):
    def __init__(self, data_dir, labels):
        super().__init__(data_dir,
                         labels,
                         EncoderAvg(data_dir, input_limit=300),
                         DecoderBR(data_dir, len(labels), [1024, 1024]),
                         name="avg_br")


class MAXxBR(Trainer):
    def __init__(self, data_dir, labels):
        super().__init__(data_dir,
                         labels,
                         EncoderMax(data_dir, input_limit=300),
                         DecoderBR(data_dir, len(labels), [1024, 1024]),
                         name="max_br")


class RNNlastxBR(Trainer):
    def __init__(self, data_dir, labels):
        super().__init__(data_dir,
                         labels,
                         EncoderRNNLast(data_dir, input_limit=300),
                         DecoderBR(data_dir, len(labels), [1024, 1024]),
                         name="rnn_last_br")


class RNNavgxBR(Trainer):
    def __init__(self, data_dir, labels):
        super().__init__(data_dir,
                         labels,
                         EncoderRNNAvg(data_dir, input_limit=300),
                         DecoderBR(data_dir, len(labels), [1024, 1024]),
                         name="rnn_avg_br")


class CNNxBR(Trainer):
    def __init__(self, data_dir, labels):
        super().__init__(data_dir,
                         labels,
                         EncoderCNN(data_dir),
                         DecoderBR(data_dir, len(labels), [1024, 1024]),
                         name="cnn_br",
                         batch_size=16)
