"""Diplays a simple menu to download datasets."""

import os
import sys
import anna.data.dataset.aida
import anna.data.dataset.ud
import anna.data.dataset.conll03
import anna.data.dataset.wmt
import anna.data.dataset.wiki
import anna.data.dataset.rcv1
import anna.data.dataset.reuters21578

options = [
    {
        "name": "Universal Dependencies (Morph, POS Tags)",
        "fetcher": anna.data.dataset.ud.fetch
    },
    {
        "name": "CoNLL03 (NER)",
        "fetcher": anna.data.dataset.conll03.fetch
    },
    {
        "name": "WMT14 (MT)",
        "fetcher": anna.data.dataset.wmt.fetch
    },
    {
        "name": "AIDA (Entity Linking)",
        "fetcher": anna.data.dataset.aida.fetch
    },
    {
        "name": "RCV1 (Multi-label Classification)",
        "fetcher": anna.data.dataset.rcv1.fetch
    },
    {
        "name": "Reuters-21578 (Multi-label Classification)",
        "fetcher": anna.data.dataset.reuters21578.fetch
    },
    {
        "name": "Wiki",
        "fetcher": anna.data.dataset.wiki.fetch
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
