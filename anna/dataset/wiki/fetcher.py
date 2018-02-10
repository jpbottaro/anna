"""Fetches mediawiki dumps."""

from __future__ import division

import os
import anna.dataset.utils as utils

# An example url for "es" is: https://dumps.wikimedia.org/eswiki/20160501/
#                             eswiki-20160501-pages-articles.xml.bz2
WIKIMEDIA_HOST = "https://dumps.wikimedia.org"
DUMP_NAME = "{lang}wiki"
URL_SUFFIX = "pages-articles.xml.bz2"
FILE_NAME = DUMP_NAME + "-{date}-" + URL_SUFFIX
URL_FORMAT = WIKIMEDIA_HOST + "/" + DUMP_NAME + "/{date}/" + FILE_NAME


def fetch(data_dir, dest="wiki", langs=None, date="latest"):
    """
    Fetches dumps for all languages in 'langs', and saves them in the
    given 'folder'.

    Creates the `dest` if it doesn't exist.

    Args:
        data_dir (str): absolute path to the dir where datasets are stored
        dest (str): name for dir where the wiki dumps will be extracted
        langs (list[str]): list of languages to fetch (default: ["en"])
        date (str): date of dump to fetch (default: "latest")

    Returns:
        final_dir (map[str][str]): paths for each language's dump (key is lang)
    """

    if langs is None:
        langs = ['en']

    # Create folder
    wiki_dir = os.path.join(data_dir, dest)
    utils.create_folder(wiki_dir)

    # Download dumps for each language
    paths = {}
    for lang in langs:
        url = URL_FORMAT.format(lang=lang, date=date)
        path = os.path.join(wiki_dir, FILE_NAME.format(lang=lang, date=date))
        utils.urlretrieve(url, path)
        paths[lang] = path

    return paths
