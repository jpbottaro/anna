"""Displays a simple menu to download datasets."""

import utils.ud
import utils.conll03
import utils.wmt
import utils.wiki

DATASETS = [
    ("Universal Dependencies (Morph, POS Tags)", utils.ud.fetch),
    ("CoNLL03 (NER)", utils.conll03.fetch),
    ("WMT14 (MT)", utils.wmt.fetch),
    ("Wiki", utils.wiki.fetch)
]

if __name__ == "__main__":
    print("Select which dataset you wish to fetch:\n")
    for i, t in enumerate(DATASETS):
        print("[{}]: {}".format(i, t[0]))
    num = input("\nSelect number: ")
    DATASETS[int(num)][1]()
