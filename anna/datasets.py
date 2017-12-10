"""Diplays a simple menu to download datasets."""

import os
import sys
import dataset.aida.fetcher
import dataset.ud.fetcher
import dataset.conll03.fetcher
import dataset.wmt.fetcher
import dataset.wiki.fetcher
import dataset.rcv1.fetcher
import dataset.reuters21578.fetcher

options = [
    {
        "name": "Universal Dependencies (Morph, POS Tags)",
        "fetcher": dataset.ud.fetcher.fetch
    },
    {
        "name": "CoNLL03 (NER)",
        "fetcher": dataset.conll03.fetcher.fetch
    },
    {
        "name": "WMT14 (MT)",
        "fetcher": dataset.wmt.fetcher.fetch
    },
    {
        "name": "AIDA (Entity Linking)",
        "fetcher": dataset.aida.fetcher.fetch
    },
    {
        "name": "RCV1 (Multi-label Classification)",
        "fetcher": dataset.rcv1.fetcher.fetch
    },
    {
        "name": "Reuters-21578 (Multi-label Classification)",
        "fetcher": dataset.reuters21578.fetcher.fetch
    },
    {
        "name": "Wiki",
        "fetcher": dataset.wiki.fetcher.fetch
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
