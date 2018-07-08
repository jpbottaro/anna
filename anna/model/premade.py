from anna.model.trainer import Trainer
from anna.model.encode import *
from anna.model.decode import *


class AVGxBR(Trainer):
    def __init__(self,
                 data_dir,
                 labels,
                 layers=2,
                 hidden_size=2048,
                 dropout=.5,
                 pretrained_embeddings="glove",
                 name="avg_br",
                 *args,
                 **kwargs):
        super().__init__(data_dir,
                         labels,
                         EncoderAvg(data_dir,
                                    pretrained_embeddings=pretrained_embeddings,
                                    input_limit=300),
                         DecoderBR(data_dir,
                                   len(labels),
                                   [hidden_size] * layers,
                                   dropout=dropout),
                         name=name,
                         *args, **kwargs)

    def __repr__(self):
        return "AVGxBR - Average input, independent binary predictions."


class MAXxBR(Trainer):
    def __init__(self,
                 data_dir,
                 labels,
                 layers=2,
                 hidden_size=2048,
                 dropout=.5,
                 name="max_br",
                 *args,
                 **kwargs):
        super().__init__(data_dir,
                         labels,
                         EncoderMax(data_dir, input_limit=300),
                         DecoderBR(data_dir,
                                   len(labels),
                                   [hidden_size] * layers,
                                   dropout=dropout),
                         name=name,
                         *args, **kwargs)

    def __repr__(self):
        return "MAXxBR - Max pooling for the input, " + \
               "independent binary predictions."


class CNNxBR(Trainer):
    def __init__(self,
                 data_dir,
                 labels,
                 layers=2,
                 hidden_size=2048,
                 dropout=.5,
                 name="cnn_br",
                 *args,
                 **kwargs):
        super().__init__(data_dir,
                         labels,
                         EncoderCNN(data_dir),
                         DecoderBR(data_dir,
                                   len(labels),
                                   [hidden_size] * layers,
                                   dropout=dropout),
                         name=name,
                         batch_size=16,
                         *args, **kwargs)

    def __repr__(self):
        return "CNNxBR - Convolutional network to analyze input, " + \
               "independent binary predictions."


class AVGxRNN(Trainer):
    def __init__(self, data_dir, labels, name="avg_rnn", *args, **kwargs):
        super().__init__(data_dir,
                         labels,
                         EncoderAvg(data_dir, input_limit=300),
                         DecoderRNN(data_dir, labels, beam_width=12),
                         name=name,
                         grad_clip=5.0,
                         *args, **kwargs)

    def __repr__(self):
        return "AVGxRNN - Average input, sequence prediction with a GRU."


class RNNxBR(Trainer):
    def __init__(self, data_dir, labels, name="rnn_br", *args, **kwargs):
        super().__init__(data_dir,
                         labels,
                         EncoderBiRNN(data_dir, input_limit=300),
                         DecoderBR(data_dir, len(labels), [2048, 2048]),
                         name=name,
                         grad_clip=5.0,
                         *args, **kwargs)

    def __repr__(self):
        return "RNNxBR - Recurrent network to analyze input, " + \
               "independent binary predictions."


class RNNxRNN(Trainer):
    def __init__(self, data_dir, labels, beam_width=12, name="rnn_rnn",
                 *args, **kwargs):
        super().__init__(data_dir,
                         labels,
                         EncoderBiRNN(data_dir, input_limit=300),
                         DecoderRNN(data_dir, labels, beam_width=beam_width),
                         name=name,
                         learning_rate=0.0002,
                         grad_clip=5.0,
                         *args, **kwargs)

    def __repr__(self):
        return "RNNxRNN - Recurrent network to analyze input, " + \
               "sequence prediction with a GRU."


class EncDec(Trainer):
    def __init__(self, data_dir, labels, beam_width=12, name="enc_dec",
                 *args, **kwargs):
        super().__init__(data_dir,
                         labels,
                         EncoderBiRNN(data_dir, input_limit=300),
                         DecoderRNN(data_dir, labels,
                                    attention=tf.contrib.seq2seq.LuongAttention,
                                    beam_width=beam_width),
                         name=name,
                         learning_rate=0.0002,
                         grad_clip=5.0,
                         *args, **kwargs)

    def __repr__(self):
        return "EncDec - Recurrent network to analyze input, " + \
               "sequence prediction with RNN and attention (GRU & Luong)."


BEST = [AVGxBR, AVGxRNN, EncDec]
CANDIDATES = [AVGxBR, AVGxRNN, RNNxBR, RNNxRNN, EncDec]
ALL = [AVGxBR, MAXxBR, CNNxBR, RNNxBR, AVGxRNN, RNNxRNN, EncDec]
