"""Fetches the Universal Dependencies.

Visit: http://universaldependencies.org"""

import os
import anna.dataset.utils as utils

UD_URL = "https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/"

UD_LINKS = {
    "1.0": UD_URL + "1-1464/universal-dependencies-1.0.tgz",
    "1.1": UD_URL + "LRT-1478/ud-treebanks-v1.1.tgz",
    "1.2": UD_URL + "1-1548/ud-treebanks-v1.2.tgz",
    "1.3": UD_URL + "1-1699/ud-treebanks-v1.3.tgz",
    "1.4": UD_URL + "1-1827/ud-treebanks-v1.4.tgz"
}


def fetch(data_dir, dest="universal-dependencies", versions=None):
    """
    Fetches and extracts the requested versions of the universal
    dependencies, and saves them in the given 'folder'.

    Creates the `dest` if it doesn't exist.

    Args:
        data_dir (str): absolute path to the folder where datasets are stored
        dest (str): name for dir where UD will be extracted
        versions (list[str]): list of UD versions to fetch

    Returns:
        final_dir (str): absolute path where UD was extracted
    """

    # Create folder
    ud_dir = os.path.join(data_dir, dest)
    utils.create_folder(ud_dir)

    if versions is None:
        versions = ["1.4"]

    for ver in versions:
        if ver not in UD_LINKS:
            print("Version not supported: " + ver)
        url = UD_LINKS[ver]
        path = os.path.join(ud_dir, "ud-" + ver + ".tgz")
        if not os.path.exists(path):
            utils.urlretrieve(url, path)

    return ud_dir
