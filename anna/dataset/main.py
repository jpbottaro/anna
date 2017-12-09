"""Displays a simple menu to download datasets."""

import os
import fetcher.aida
import fetcher.ud
import fetcher.conll03
import fetcher.wmt
import fetcher.wiki
import fetcher.reuters21578

datasets = [
    {
        "name": "Universal Dependencies (Morph, POS Tags)",
        "fetcher": fetcher.ud.fetch,
        "folder": "universal-dependencies"
    },
    {
        "name": "CoNLL03 (NER)",
        "fetcher": fetcher.conll03.fetch,
        "folder": "conll03"
    },
    {
        "name": "WMT14 (MT)",
        "fetcher": fetcher.wmt.fetch,
        "folder": "wmt14"
    },
    {
        "name": "AIDA (Entity Linking)",
        "fetcher": fetcher.aida.fetch,
        "folder": "aida"
    },
    {
        "name": "Reuters-21578 (Multi-label Classification)",
        "fetcher": fetcher.reuters21578.fetch,
        "folder": "reuters-21578"
    },
    {
        "name": "Wiki",
        "fetcher": fetcher.wiki.fetch,
        "folder": "wiki"
    }
]

if __name__ == "__main__":
    print("Select which dataset you wish to fetch:\n")
    for i, t in enumerate(datasets):
        print("[{}]: {}".format(i, t["name"]))
    num = input("\nSelect number: ")
    if num:
        print()
        dataset = datasets[int(num)]
        datasets_folder = os.path.dirname(os.path.realpath(__file__))
        datasets_folder = os.path.join(datasets_folder, "data")
        datasets_folder = os.path.join(datasets_folder, dataset["folder"])
        dataset["fetcher"](datasets_folder)
