"""Displays a simple menu to download datasets."""

import os
import fetchers.aida
import fetchers.ud
import fetchers.conll03
import fetchers.wmt
import fetchers.wiki

DATASETS = [
    ("Universal Dependencies (Morph, POS Tags)", fetchers.ud.fetch),
    ("CoNLL03 (NER)", fetchers.conll03.fetch),
    ("WMT14 (MT)", fetchers.wmt.fetch),
    ("AIDA (Entity Linking)", fetchers.aida.fetch),
    ("Wiki", fetchers.wiki.fetch)
]

if __name__ == "__main__":
    print("Select which dataset you wish to fetch:\n")
    for i, t in enumerate(DATASETS):
        print("[{}]: {}".format(i, t[0]))
    num = input("\nSelect number: ")
    if num:
        print()
        datasets_folder = os.path.dirname(os.path.realpath(__file__))
        datasets_folder = os.path.join(datasets_folder, "data")
        DATASETS[int(num)][1](datasets_folder)
