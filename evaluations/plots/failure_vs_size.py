"""Round-trip failure rate vs molecule size, PubChem + ZINC22 combined.

Pools all ten 100k seed subsets (5 PubChem + 5 ZINC22 = 1M molecules) into one
failure-rate-vs-heavy-atom-count curve per model. A molecule is a "failure" when
its IUPAC name does not round-trip via OPSIN + standardization (appears in the
``*_opsin_failures.csv``). Per-bin failure rate is computed for each of the ten
100k subsets; markers show the mean and error bars the std across those subsets
(clipped at 0 -- a rate can't be negative). Bins with too few molecules in a
subset are dropped so the tail is not noise.

Heavy-atom counting for the full 1M molecules runs across CPU cores.

    python plots/failure_vs_size.py               # writes failure_vs_size_combined.png/.pdf
"""

import csv
import glob
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import lama_aesthetics
from lama_aesthetics.plotutils import range_frame

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
DATASETS = ["pubchem", "zinc22"]
METHODS = [("Openclatura", "openclatura"), ("STOUT", "stout")]

BIN_EDGES = np.array([0, 10, 15, 20, 25, 30, 35, 40, 50, 200])
BIN_CENTERS = 0.5 * (BIN_EDGES[:-1] + BIN_EDGES[1:])
BIN_CENTERS[-1] = (BIN_EDGES[-2] + BIN_EDGES[-2] + 15) / 2  # clamp open-ended center
MIN_BIN = 50   # ignore a bin in a subset if it holds fewer molecules than this
WORKERS = 64


# --------------------------------------------------------------------------- #
# multicore heavy-atom counting
# --------------------------------------------------------------------------- #
def _init():
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")


def _hac_chunk(smiles):
    from rdkit import Chem
    out = []
    for s in smiles:
        m = Chem.MolFromSmiles(s) if s else None
        out.append(m.GetNumHeavyAtoms() if m is not None else -1)
    return out


def parallel_hac(smiles, chunk=5000):
    chunks = [smiles[i : i + chunk] for i in range(0, len(smiles), chunk)]
    result = []
    with ProcessPoolExecutor(max_workers=WORKERS, initializer=_init) as ex:
        for part in ex.map(_hac_chunk, chunks):
            result.extend(part)
    return np.array(result)


# --------------------------------------------------------------------------- #
def load_subsets():
    """Return list of dicts: {name, indices, smiles} for each 100k subset."""
    subsets = []
    for ds in DATASETS:
        for pred in sorted(glob.glob(os.path.join(RESULTS, ds, "*_openclatura.jsonl"))):
            idx, smi = [], []
            with open(pred) as f:
                for line in f:
                    r = json.loads(line)
                    idx.append(r["index"])
                    smi.append(r["smiles"])
            stem = os.path.basename(pred).replace("_openclatura.jsonl", "")
            subsets.append({"ds": ds, "stem": stem, "indices": np.array(idx), "smiles": smi})
    return subsets


def failed_indices(ds, stem, model):
    fp = os.path.join(RESULTS, ds, f"{stem}_{model}_opsin_failures.csv")
    failed = set()
    with open(fp, newline="") as f:
        for row in csv.DictReader(f):
            failed.add(int(row["index"]))
    return failed


def per_bin_rates(hac, is_failure):
    """Failure rate (%) per bin for one subset; nan where < MIN_BIN molecules."""
    valid = hac >= 0
    bin_idx = np.digitize(hac, BIN_EDGES[1:-1])
    rates = np.full(len(BIN_CENTERS), np.nan)
    for b in range(len(BIN_CENTERS)):
        mask = valid & (bin_idx == b)
        if mask.sum() >= MIN_BIN:
            rates[b] = is_failure[mask].mean() * 100.0
    return rates


def main():
    subsets = load_subsets()
    print(f"{len(subsets)} subsets, {sum(len(s['smiles']) for s in subsets):,} molecules total")

    # one big multicore heavy-atom-count pass over all molecules
    flat = [s for sub in subsets for s in sub["smiles"]]
    hac_all = parallel_hac(flat)
    off = 0
    for sub in subsets:
        n = len(sub["smiles"])
        sub["hac"] = hac_all[off : off + n]
        off += n

    curves = []
    for label, model in METHODS:
        per_subset = []
        for sub in subsets:
            failed = failed_indices(sub["ds"], sub["stem"], model)
            is_fail = np.array([i in failed for i in sub["indices"]], dtype=bool)
            per_subset.append(per_bin_rates(sub["hac"], is_fail))
        per_subset = np.vstack(per_subset)
        with np.errstate(invalid="ignore"):
            mean = np.nanmean(per_subset, axis=0)
            std = np.nanstd(per_subset, axis=0)
        # require >= 3 subsets contributing to a bin, else drop it
        n_valid = np.sum(~np.isnan(per_subset), axis=0)
        mean[n_valid < 3] = np.nan
        curves.append((label, mean, std))

    lama_aesthetics.get_style("main")
    mpl.rcParams.update({
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    })
    fig, ax = plt.subplots(
        figsize=(lama_aesthetics.ONE_COL_WIDTH, lama_aesthetics.ONE_COL_HEIGHT),
    )

    all_x, all_lo, all_hi = [], [], []
    for label, mean, std in curves:
        valid = ~np.isnan(mean)
        x, y, e = BIN_CENTERS[valid], mean[valid], std[valid]
        lower = np.minimum(e, y)          # clip error bar at 0 (rate can't be < 0)
        ax.errorbar(
            x, y, yerr=[lower, e],
            marker="o", markersize=2.5, capsize=3,
            elinewidth=1.0, capthick=1.0, label=label, zorder=3,
        )
        all_x.append(x)
        all_lo.append(y - lower)
        all_hi.append(y + e)

    ax.set_xlabel("Heavy atom count")
    ax.set_ylabel("Failure rate (%)")
    range_frame(ax, np.concatenate(all_x), np.concatenate(all_lo + all_hi))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.legend()

    fig.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "failure_vs_size")
    plt.savefig(out + ".png", dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.savefig(out + ".pdf", bbox_inches="tight", pad_inches=0.01)
    print(f"saved {out}.png / .pdf")


if __name__ == "__main__":
    main()
