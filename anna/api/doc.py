"""Representation of a text document"""


class Doc:
    """A document with all extracted information (e.g. labels attached)."""

    def __init__(self, title, headline, dateline, text, labels):
        """Create a document

        Args:
            title (str): doc's title
            headline (str): doc's headline (could be None)
            dateline (str): doc's title (could be None)
            text (str): doc's text
            mentions (list[str]): labels attached to the doc
        """
        self.title = title
        self.headline = headline
        self.dateline = dateline
        self.text = text
        self.labels = labels

    def __str__(self):
        return self.title
