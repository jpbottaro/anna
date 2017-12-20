"""Standard utilities to work with text"""

import copy
from tensorflow.python.keras.preprocessing.sequence import pad_sequences


def tokenize(text):
    """
    Tokenizes the given `text`.

    Args:
        text (str): a piece of text to tokenize

    Returns:
        tokens (list[str]): list of tokens from `text`
    """
    if not text:
        return []

    tokens = []
    split = text.split()
    for s in split:
        s = s.strip()
        if s == "":
            continue

        # Separate leading punctuation
        if len(s) > 1:
            if s[0] == "(" or s[0] == "\"" or s[0] == "'":
                tokens.append(s[0])
                s = s[1:]

        # Separate contractions
        apos_index = s.find("'")
        while apos_index > 0 and apos_index < len(s):
            tokens.append(s[:apos_index])
            s = s[apos_index:]
            apos_index = s.find("'")

        # Separate "/"
        parts = s.split("/")
        for part in parts[:-1]:
            tokens.append(part)
            tokens.append("/")
        s = parts[-1]

        # Separate ending punctuation
        if len(s) > 1:
            if (s[-1] == "." and s[-2].islower()) \
                    or s[-1] == "," \
                    or s[-1] == ")" \
                    or s[-1] == "'" \
                    or s[-1] == "\"" \
                    or s[-1] == ";":
                tokens.append(s[:-1])
                s = s[-1]

        tokens.append(s)
    return tokens


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
    return pad_sequences(ids, maxlen=max_steps)


def clean(docs):
    """
    Returns a new set of documents like `docs`, but without the labels.

    Args:
        docs (list[Doc]): list of document to clean

    Returns:
        analyzed_docs (list[Doc]): same as `docs`, without labels
    """
    new_docs = [copy.copy(d) for d in docs]
    for doc in new_docs:
        doc.labels = []
    return new_docs
