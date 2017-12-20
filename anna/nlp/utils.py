"""Standard utilities to work with text"""

import copy


def preprocess(docs, vocabulary):
    """
    Applies common information extraction techniques to a doc, like
    tokenization, stemming, and more.

    Args:
        docs (list[Doc]): list of document to analyze
        vocabulary (dict): mapping of word to id, and id to word

    Returns:
        analyzed_docs (list[Doc]): same as `docs`, with added analysis
    """
    docs = [tokenize_doc(doc) for doc in docs]
    docs = [apply_vocabulary(doc, vocabulary) for doc in docs]
    return docs


def tokenize_doc(doc):
    """
    Tokenizes all fields in a `Doc`.

    Args:
        doc (Doc): document to tokenize

    Returns:
        tokenized_doc (Doc): same as `doc`, with added tokenization
    """
    doc.title_tokens = tokenize(doc.title)
    doc.headline_tokens = tokenize(doc.headline)
    doc.dateline_tokens = tokenize(doc.dateline)
    doc.text_tokens = tokenize(doc.text)
    doc.labels_tokens = [tokenize(l) for l in doc.labels]
    return doc


def tokenize(text):
    """
    Tokenizes the given `text`.

    Args:
        text (str): a piece of text to tokenize

    Returns:
        tokens (list[str]): list of tokens from `text`
    """
    if not text:
        return None

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


def apply_vocabulary(doc, voc):
    """
    Matches all tokens in `Doc` with their appropriate id in the vocabulary.

    Args:
        doc (Doc): a tokenized document

    Returns:
        analyzed_doc (Doc): same as `doc`, with the added ids from `voc`
    """
    def map_ids(tokens):
        if not tokens:
            return None
        return [voc[t] for t in tokens]

    doc.title_ids = map_ids(doc.title_tokens)
    doc.headline_ids = map_ids(doc.headline_tokens)
    doc.dateline_ids = map_ids(doc.dateline_tokens)
    doc.text_ids = map_ids(doc.text_tokens)
    doc.labels_ids = [map_ids(tokens) for tokens in doc.labels_tokens]
    return doc


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
