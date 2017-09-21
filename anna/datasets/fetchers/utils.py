"""Simple utilities to fetch datasets."""

from __future__ import division
import os
import sys
import urllib.request


def reporthook(blocknum, blocksize, totalsize):
    """
    A hook that conforms with 'urllib.request.urlretrieve()' interface.

    It reports in stdout the current progress of the download, including
    a progress bar.
    """

    readsofar = blocknum * blocksize
    if totalsize > 0:
        percent = readsofar * 100 / totalsize
        done = int(50 * percent / 100)
        s = "\r{:05.2f}% [{}{}] {:.2f}mb".format(percent,
                                                 '=' * done,
                                                 ' ' * (50 - done),
                                                 totalsize / 2 ** 20)
        sys.stdout.write(s)
        if readsofar >= totalsize:
            sys.stdout.write("\n")
        sys.stdout.flush()
    # total size is unknown
    else:
        sys.stdout.write("read {:.2f}mb\n".format(readsofar / 2 ** 20))


def urlretrieve(url, path):
    """
    Same as 'urllib.urlretrieve()', but with a nice reporthook to show
    a progress bar.

    If 'path' exists, doesn't download anything.
    """
    if os.path.exists(path):
        print("Skipping: " + url)
    else:
        print("Downloading: " + url)
        urllib.request.urlretrieve(url, path, reporthook)


def create_folder(folder):
    """
    Creates the given folder in the filesystem.

    The whole path is created, just like 'mkdir -p' would do.
    """

    try:
        os.makedirs(folder)
    except OSError:
        if not os.path.isdir(folder):
            raise
