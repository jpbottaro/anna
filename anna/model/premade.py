from anna.model.trainer import Trainer
from anna.model.encode import *
from anna.model.decode import *


class AVGxBR(Trainer):
    def __init__(self, model_dir, labels, words, embeddings):
        super().__init__(model_dir,
                         labels,
                         EncoderAvg(model_dir, words, embeddings, max_size=300),
                         DecoderBR(model_dir, len(labels), [1024, 512, 512]))


class RNNxBR(Trainer):
    def __init__(self, model_dir, labels, words, embeddings):
        super().__init__(model_dir,
                         labels,
                         EncoderRNN(model_dir, words, embeddings, max_size=300),
                         DecoderBR(model_dir, len(labels), [1024, 512, 512]))


class CNNxBR(Trainer):
    def __init__(self, model_dir, labels, words, embeddings):
        super().__init__(model_dir,
                         labels,
                         EncoderCNN(model_dir, words, embeddings),
                         DecoderBR(model_dir, len(labels), [1024, 512, 512]),
                         batch_size=8)
