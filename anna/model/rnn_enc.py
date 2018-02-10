"""MLC model using RNNs for encoding."""

import os
import tensorflow as tf
from anna.model.trainer import Trainer
from anna.model.encoder.rnn import RNNEncoder
from anna.model.decoder.feedforward import FeedForwardDecoder


class EncRNN(Trainer):
    """
    Maps a Multi-label classification problem independeny binary classifiers,
    with an RNN encoding the input sequence.
    """

    def __init__(self,
                 data_dir,
                 labels,
                 name=None,
                 optimizer="adam",
                 metric="val_ebf1",
                 max_words=300,
                 hidden_size=1024,
                 voc_size=300000,
                 fixed_emb=False,
                 save=False,
                 verbose=True):
        """
        Creates an RNN-based model for MLC. The RNN is only used for encoding.

        Args:
            data_dir (str): path to the folder where datasets are stored
            labels (list[str]): list of possible outputs
            name (str): name of the model (used when serializing to disk)
                        (default: combination of parameters)
            optimizer (str): one of: adam, rmsprop, momentum (default: adam)
            metric (str): metric to optimize (e.g. val_loss, val_acc, etc.)
                          (default: val_ebf1)
            max_words (int): number of words to use when embedding text fields
                             (default: 300)
            hidden_size (int): size of the hidden units on each hidden layer
                               (default: 1024)
            voc_size (int): maximum size of the word vocabulary
                            (default: 300000)
            fixed_emb (bool): True if word embeddings shouldn't be trainable
                              (default: False)
            save (bool): always save the best model (default: False)
            verbose (bool): print messages of progress (default: True)
        """
        # Encode doc using an RNN, producing a fixed output (concatenation
        # of the final outputs from the forward/backwards passes)
        encoder = RNNEncoder(data_dir, hidden_size, max_words, fixed_emb,
                             voc_size)

        # Classify labels with independent logistic regressions
        decoder = FeedForwardDecoder(data_dir, labels,
                                     num_layers=2,
                                     hidden_size=hidden_size,
                                     chain=False,
                                     hinge=False)

        # Generate name
        if not name:
            name = "encrnn_{}_voc-{}_hidden-{}{}"
            name = name.format(optimizer, voc_size, hidden_size,
                               "_fixed-emb" if fixed_emb else "")

        super().__init__(data_dir, labels, name, encoder, decoder, optimizer,
                         metric, save, verbose)
