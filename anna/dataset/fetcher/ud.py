"""Fetches the Universal Dependencies.

Visit: http://universaldependencies.org"""

import os
from . import utils

UD_URL = "https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/"

UD_LINKS = {
    "1.0": UD_URL + "1-1464/universal-dependencies-1.0.tgz",
    "1.1": UD_URL + "LRT-1478/ud-treebanks-v1.1.tgz",
    "1.2": UD_URL + "1-1548/ud-treebanks-v1.2.tgz",
    "1.3": UD_URL + "1-1699/ud-treebanks-v1.3.tgz",
    "1.4": UD_URL + "1-1827/ud-treebanks-v1.4.tgz"
}


def fetch(folder, versions=None):
    """
    Fetches and extracts the requested versions of the universal
    dependencies, and saves them in the given 'folder'.

    Creates the folder if it doesn't exist.
    """
    if versions is None:
        versions = ["1.4"]

    utils.create_folder(folder)
    paths = []
    for ver in versions:
        if ver not in UD_LINKS:
            print("Version not supported: " + ver)
        url = UD_LINKS[ver]
        path = os.path.join(folder, "ud-" + ver + ".tgz")
        paths.append((ver, path))
        utils.urlretrieve(url, path)
    return paths
