"""Per-molecule inference time vs molecule size (heavy atom count).

Times STOUT (local STOUT-pypi-2.0.5 translate_forward) and Openclatura
(openclatura.name_smiles) one molecule at a time on a 200-molecule PubChem
sample, and scatters time-per-molecule against heavy-atom count. Single panel
(the earlier right-hand 10k-molecule histogram is intentionally dropped).

    CUDA_VISIBLE_DEVICES=1 python plots/time_vs_size.py
"""

import json
import os
import random
import time

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from rdkit import Chem, RDLogger
import lama_aesthetics
from lama_aesthetics.plotutils import range_frame

from STOUT import translate_forward          # local modified STOUT-pypi-2.0.5
from openclatura import name_smiles

RDLogger.DisableLog("rdApp.*")

K = 200
DATA = os.path.join(os.path.dirname(__file__), "..", "data", "pubchem", "pubchem_seed42_100000_input.jsonl")


def load_sample():
    rows = [json.loads(l) for l in open(DATA)]
    random.seed(0)
    random.shuffle(rows)
    smiles, sizes = [], []
    for r in rows:
        m = Chem.MolFromSmiles(r["smiles"])
        if m is None:
            continue
        smiles.append(r["smiles"])
        sizes.append(m.GetNumHeavyAtoms())
        if len(smiles) >= K:
            break
    return smiles, sizes


def main():
    smiles, sizes = load_sample()
    print(f"{len(smiles)} molecules")

    # warmup (loads models / JIT; discard)
    translate_forward(smiles[0])
    name_smiles(smiles[0])

    stout_times, oc_times = [], []
    for i, smi in enumerate(smiles):
        t = time.perf_counter()
        translate_forward(smi)
        stout_times.append(time.perf_counter() - t)

        t = time.perf_counter()
        name_smiles(smi)
        oc_times.append(time.perf_counter() - t)
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(smiles)}")

    print(f"STOUT       median {np.median(stout_times)*1e3:.1f} ms")
    print(f"Openclatura median {np.median(oc_times)*1e3:.1f} ms")

    lama_aesthetics.get_style("main")
    mpl.rcParams.update({
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    })
    fig, ax = plt.subplots(
        figsize=(lama_aesthetics.ONE_COL_WIDTH, lama_aesthetics.ONE_COL_HEIGHT),
    )

    ax.scatter(sizes, stout_times, label="STOUT", alpha=0.7)
    ax.scatter(sizes, oc_times, label="Openclatura", alpha=0.7)
    ax.set_xlabel("Heavy atom count")
    ax.set_ylabel("Time per molecule (s)")

    all_sizes = np.array(sizes + sizes)
    all_times = np.array(stout_times + oc_times)
    range_frame(ax, all_sizes, all_times)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.legend()

    fig.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "time_vs_size")
    plt.savefig(out + ".png", dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.savefig(out + ".pdf", bbox_inches="tight", pad_inches=0.01)
    print(f"saved {out}.png / .pdf")


if __name__ == "__main__":
    main()
