from anna.model.trainer import Trainer
from anna.model.encode import *
from anna.model.decode import *


class AVGxBR(Trainer):
    def __init__(self, data_folder, labels, vocab, embeddings):
        super().__init__(data_folder,
                         labels,
                         EncoderAvg(vocab, embeddings, max_size=300),
                         DecoderBR(len(labels), [1024, 512, 512]))


class RNNxBR(Trainer):
    def __init__(self, data_folder, labels, vocab, embeddings):
        super().__init__(data_folder,
                         labels,
                         EncoderRNN(vocab, embeddings, max_size=300),
                         DecoderBR(len(labels), [1024, 512, 512]))


class CNNxBR(Trainer):
    def __init__(self, data_folder, labels, vocab, embeddings):
        super().__init__(data_folder,
                         labels,
                         EncoderCNN(vocab, embeddings),
                         DecoderBR(len(labels), [1024, 512, 512]),
                         batch_size=8)
