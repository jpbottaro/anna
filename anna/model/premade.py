from tensorflow.contrib.seq2seq import LuongAttention
from anna.model.trainer import Trainer
from anna.model.encode import *
from anna.model.decode import *


class AVGxBR(Trainer):
    def __init__(self, data_dir, labels, *args, **kwargs):
        super().__init__(data_dir,
                         labels,
                         EncoderAvg(data_dir, input_limit=300),
                         DecoderBR(data_dir, len(labels), [1024, 1024]),
                         name="avg_br",
                         *args, **kwargs)

    def __repr__(self):
        return "AVGxBR - Average input, independent binary predictions."


class MAXxBR(Trainer):
    def __init__(self, data_dir, labels, *args, **kwargs):
        super().__init__(data_dir,
                         labels,
                         EncoderMax(data_dir, input_limit=300),
                         DecoderBR(data_dir, len(labels), [1024, 1024]),
                         name="max_br",
                         *args, **kwargs)

    def __repr__(self):
        return "MAXxBR - Max pooling for the input, " + \
               "independent binary predictions."


class CNNxBR(Trainer):
    def __init__(self, data_dir, labels, *args, **kwargs):
        super().__init__(data_dir,
                         labels,
                         EncoderCNN(data_dir),
                         DecoderBR(data_dir, len(labels), [1024, 1024]),
                         name="cnn_br",
                         batch_size=16,
                         *args, **kwargs)

    def __repr__(self):
        return "CNNxBR - Convolutional network to analyze input, " + \
               "independent binary predictions."


class RNNxBR(Trainer):
    def __init__(self, data_dir, labels, *args, **kwargs):
        super().__init__(data_dir,
                         labels,
                         EncoderBiRNN(data_dir, input_limit=300),
                         DecoderBR(data_dir, len(labels), [1024, 1024]),
                         name="rnn_br",
                         *args, **kwargs)

    def __repr__(self):
        return "RNNxBR - Recurrent network to analyze input, " + \
               "independent binary predictions."


class AVGxRNN(Trainer):
    def __init__(self, data_dir, labels, *args, **kwargs):
        super().__init__(data_dir,
                         labels,
                         EncoderAvg(data_dir, input_limit=300),
                         DecoderRNN(data_dir, labels, beam_width=12),
                         name="avg_rnn",
                         learning_rate=0.001,
                         grad_clip=1.0,
                         *args, **kwargs)

    def __repr__(self):
        return "AVGxRNN - Average input, sequence prediction with RNN (GRU)."


class RNNxRNN(Trainer):
    def __init__(self, data_dir, labels, *args, **kwargs):
        super().__init__(data_dir,
                         labels,
                         EncoderBiRNN(data_dir, input_limit=300),
                         DecoderRNN(data_dir, labels, beam_width=12),
                         name="rnn_rnn",
                         learning_rate=0.001,
                         grad_clip=1.0,
                         *args, **kwargs)

    def __repr__(self):
        return "RNNxRNN - Recurrent network to analyze input, " + \
               "sequence prediction with RNN (GRU)."


class EncDec(Trainer):
    def __init__(self, data_dir, labels, *args, **kwargs):
        super().__init__(data_dir,
                         labels,
                         EncoderBiRNN(data_dir, input_limit=300),
                         DecoderAttRNN(data_dir, labels, beam_width=12),
                         name="enc_dec",
                         learning_rate=0.001,
                         grad_clip=1.0,
                         *args, **kwargs)

    def __repr__(self):
        return "EncDec - Recurrent network to analyze input, " + \
               "sequence prediction with RNN and attention (GRU & Bahdanau)."


class EncDecLuong(Trainer):
    def __init__(self, data_dir, labels, *args, **kwargs):
        super().__init__(data_dir,
                         labels,
                         EncoderBiRNN(data_dir, input_limit=300),
                         DecoderAttRNN(data_dir,
                                       labels,
                                       attention=LuongAttention,
                                       beam_width=12),
                         name="enc_dec_luong",
                         learning_rate=0.001,
                         grad_clip=1.0,
                         *args, **kwargs)

    def __repr__(self):
        return "EncDecLuong - Recurrent network to analyze input, " + \
               "sequence prediction with RNN and attention (GRU & Luong)."


ALL = [AVGxBR, MAXxBR, CNNxBR, RNNxBR, AVGxRNN, RNNxRNN, EncDec, EncDecLuong]
