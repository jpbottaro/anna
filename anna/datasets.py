"""Diplays a simple menu to download datasets."""

import os
import sys
import anna.dataset.aida.fetcher
import anna.dataset.ud.fetcher
import anna.dataset.conll03.fetcher
import anna.dataset.wmt.fetcher
import anna.dataset.wiki.fetcher
import anna.dataset.rcv1.fetcher
import anna.dataset.reuters21578.fetcher

options = [
    {
        "name": "Universal Dependencies (Morph, POS Tags)",
        "fetcher": anna.dataset.ud.fetcher.fetch
    },
    {
        "name": "CoNLL03 (NER)",
        "fetcher": anna.dataset.conll03.fetcher.fetch
    },
    {
        "name": "WMT14 (MT)",
        "fetcher": anna.dataset.wmt.fetcher.fetch
    },
    {
        "name": "AIDA (Entity Linking)",
        "fetcher": anna.dataset.aida.fetcher.fetch
    },
    {
        "name": "RCV1 (Multi-label Classification)",
        "fetcher": anna.dataset.rcv1.fetcher.fetch
    },
    {
        "name": "Reuters-21578 (Multi-label Classification)",
        "fetcher": anna.dataset.reuters21578.fetcher.fetch
    },
    {
        "name": "Wiki",
        "fetcher": anna.dataset.wiki.fetcher.fetch
    }
]

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: main.py DATA_DIR")
        exit(1)

    print("Select which dataset you wish to fetch:\n")
    for i, t in enumerate(options):
        print("[{}]: {}".format(i, t["name"]))

    num = input("\nSelect number: ")
    if num:
        print()
        data = options[int(num)]
        datasets_dir = os.path.abspath(sys.argv[1])
        data["fetcher"](datasets_dir)
