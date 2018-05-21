from anna.model.trainer import Trainer
from anna.model.encode import *
from anna.model.decode import *
from anna.model.bridge import *


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


class CNNxBR(Trainer):
    def __init__(self, data_dir, labels):
        super().__init__(data_dir,
                         labels,
                         EncoderCNN(data_dir),
                         DecoderBR(data_dir, len(labels), [1024, 1024]),
                         name="cnn_br",
                         batch_size=16)


class RNNxBR(Trainer):
    def __init__(self, data_dir, labels):
        super().__init__(data_dir,
                         labels,
                         EncoderBiRNN(data_dir, input_limit=300),
                         DecoderBR(data_dir, len(labels), [1024, 1024]),
                         name="rnn_br")


class EncDec(Trainer):
    def __init__(self, data_dir, labels):
        super().__init__(data_dir,
                         labels,
                         EncoderBiRNN(data_dir, input_limit=300),
                         DecoderRNN(data_dir, labels),
                         name="enc_dec",
                         learning_rate=0.00001,
                         grad_clip=1.0)


class AttEncDec(Trainer):
    def __init__(self, data_dir, labels):
        super().__init__(data_dir,
                         labels,
                         EncoderUniRNN(data_dir, input_limit=30),
                         DecoderAttRNN(data_dir, labels, bridge=ZeroBridge()),
                         name="enc_dec_att",
                         learning_rate=0.0001,
                         grad_clip=1.0)
