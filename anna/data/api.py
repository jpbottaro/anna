"""Representation of a text document"""


class Doc:
    """A document with all extracted information (e.g. labels attached)."""

    def __init__(self, doc_id, title, headline, dateline, text, labels):
        """Create a document

        Args:
            doc_id (str, optional): document id
            title (str, optional): title (can be None)
            headline (str, optional): headline (can be None)
            dateline (str, optional): doc title (can be None)
            text (str): text of the document
            labels (list[str]): labels attached to the document
        """
        self.doc_id = doc_id
        self.title = title
        self.headline = headline
        self.dateline = dateline
        self.text = text
        self.labels = labels

    def __str__(self):
        return self.title
