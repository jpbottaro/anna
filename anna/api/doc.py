"""Representation of a text document"""


class Doc:
    """A document with all extracted information (e.g. labels attached)."""

    def __init__(self, text, labels):
        """Create a document

        Args:
            text (str): doc's text.
            mentions (list[str]): labels attached to the doc.
        """
        self.text = text
        self.labels = labels
