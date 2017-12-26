"""Standard utilities to work with text"""

import re
import tensorflow as tf


# Find numbers, examples: -1 | 123 | 1.324e10 | 1,234.24
number_finder = re.compile(r"[+-]?(\d+,?)+(?:\.\d+)?(?:[eE][+-]?\d+)?")


def tokenize(text,
             remove="\"#()*+<=>@[\\]^_`{|}~\t\n",
             separate="?!/'%$&,.;:",
             number_token="1"):
    """
    Tokenizes the given `text`. Removes all tokens in `remove`, and splits
    the ones in `separate`.

    If `number_token` is not None, all numbers are modified to this token.

    Args:
        text (str): a piece of text to tokenize
        remove (str): chars that should be removed
        separate (str): chars that should separate tokens (and kept)
        number_token (str): token to use for all numbers

    Returns:
        tokens (list[str]): list of tokens from `text`
    """
    if not text:
        return []

    if number_token:
        text = number_finder.sub(number_token, text)

    remover = str.maketrans({c: " " for c in remove})
    separator = str.maketrans({c: " " + c for c in separate})
    text = text.translate(remover)
    text = text.translate(separator)

    return [t for t in text.split() if t]


def text_to_sequence(data, voc, max_steps=None):
    """
    Converts the given text to a array of indices, one for each word.

    Args:
        data (list[str]): list of text to tokenize and map to vocabulary
        voc (map[str->int]): vocabulary mapping tokens to ids
        max_steps (int): maximum size for the returned sequence

    Returns:
        seq (np.array): array with one word id per token in `text`
    """
    ids = [[voc[t] for t in tokenize(text)] for text in data]
    return tf.keras.preprocessing.sequence.pad_sequences(ids, maxlen=max_steps)
