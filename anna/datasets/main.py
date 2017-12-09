"""Displays a simple menu to download datasets."""

import os
import fetcher.aida
import fetcher.ud
import fetcher.conll03
import fetcher.wmt
import fetcher.wiki

datasets = [
    ("Universal Dependencies (Morph, POS Tags)", fetcher.ud.fetch),
    ("CoNLL03 (NER)", fetcher.conll03.fetch),
    ("WMT14 (MT)", fetcher.wmt.fetch),
    ("AIDA (Entity Linking)", fetcher.aida.fetch),
    ("Wiki", fetcher.wiki.fetch)
]

if __name__ == "__main__":
    print("Select which dataset you wish to fetch:\n")
    for i, t in enumerate(datasets):
        print("[{}]: {}".format(i, t[0]))
    num = input("\nSelect number: ")
    if num:
        print()
        datasets_folder = os.path.dirname(os.path.realpath(__file__))
        datasets_folder = os.path.join(datasets_folder, "data")
        datasets[int(num)][1](datasets_folder)
