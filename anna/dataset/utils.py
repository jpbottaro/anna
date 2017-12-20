"""Simple utilities to fetch datasets."""

import os
import sys
import pickle
import urllib.request
import numpy as np


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


class cache():
    """
    Decorator to cache the returned values of a function into pickles.

    It expects the wrapped function to recieve in the first argument a folder,
    where it will store the cached values.
    """

    def __init__(self, return_nr):
        """
        Create caching decorator for a function with `return_nr` number of
        return values.

        Args:
            return_nr (int): number of return values that should be cached
        """
        self.return_nr = return_nr

    def __call__(self, func):
        """
        Wraps a function, caching its return values into pickles.

        Args:
            func (function): function to wrap

        Returns:
            wrapper (function): wrapping function that avoids calling `func`
                                if the values were calculated before
        """
        def wrapper(*args, **kw):
            # We expect the first argument to be the data folder
            data_dir = args[0]
            saved = [(i, os.path.join(data_dir, "cache-{}".format(i)))
                     for i in range(self.return_nr)]

            # If the returned values were cached before, load them
            values = []
            for i, path in saved:
                pickle_path = path + ".pickle"
                numpy_path = path + ".npy"
                if os.path.isfile(numpy_path):
                    values.append(np.load(numpy_path))
                elif os.path.isfile(pickle_path):
                    with open(pickle_path, "rb") as f:
                        values.append(pickle.load(f))
                else:
                    break

            # If the returned values were not found, calculate and cache them
            if len(values) != len(saved):
                values = func(*args, **kw)
                for i, path in saved:
                    obj = values[i]
                    if isinstance(obj, np.ndarray):
                        np.save(path + ".npy", obj)
                    else:
                        with open(path + ".pickle", "wb") as f:
                            pickle.dump(obj, f, protocol=4)

            return values

        return wrapper
