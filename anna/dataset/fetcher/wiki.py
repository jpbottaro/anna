"""Fetches mediawiki dumps."""

from __future__ import division

import os
from . import utils

# An example url for "es" is: https://dumps.wikimedia.org/eswiki/20160501/
#                             eswiki-20160501-pages-articles.xml.bz2
WIKIMEDIA_HOST = "https://dumps.wikimedia.org"
DUMP_NAME = "{lang}wiki"
URL_SUFFIX = "pages-articles.xml.bz2"
FILE_NAME = DUMP_NAME + "-{date}-" + URL_SUFFIX
URL_FORMAT = WIKIMEDIA_HOST + "/" + DUMP_NAME + "/{date}/" + FILE_NAME


def fetch(folder, langs=None, date="latest"):
    """
    Fetches dumps for all languages in 'langs', and saves them in the
    given 'folder'.

    Creates the folder if it doesn't exist.
    """
    if langs is None:
        langs = ['en']

    utils.create_folder(folder)
    paths = []
    for lang in langs:
        url = URL_FORMAT.format(lang=lang, date=date)
        path = os.path.join(folder, FILE_NAME.format(lang=lang, date=date))
        paths.append((lang, path))
        utils.urlretrieve(url, path)
    return paths
