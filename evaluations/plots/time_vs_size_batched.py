"""Per-molecule naming time vs molecule size, STOUT batched on GPU.

Sample: for every heavy-atom count in ``HAC_RANGE``, up to ``PER_HAC`` molecules
from the PubChem seed-42 subset (random, seed 0).

* ``--part cpu``  (run in an env with openclatura + nispo, pinned to one core
  with ``taskset``): names every molecule one at a time, interleaving the two
  tools, and records the fastest of ``REPEATS`` runs per molecule.
* ``--part gpu``  (run in the STOUT env with ``CUDA_VISIBLE_DEVICES`` set):
  decodes each heavy-atom group as ONE batch with the local batched STOUT
  (``translate_forward_batch``, batch size = group size). Each group is decoded
  twice and the faster run is kept, so one-off graph tracing is not counted;
  time per molecule = batch time / group size.
* ``--part plot``: combines both into ``time_vs_size_batched.png/.pdf``
  (log y-axis, mean per heavy-atom count) and prints summary numbers.

    taskset -c 0 python plots/time_vs_size_batched.py --part cpu
    CUDA_VISIBLE_DEVICES=0 python plots/time_vs_size_batched.py --part gpu
    python plots/time_vs_size_batched.py --part plot
"""

import argparse
import json
import os
import random
import time

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data", "pubchem", "pubchem_seed42_100000_input.jsonl")
OUT_DIR = os.path.join(HERE, "timing_batched")
HAC_RANGE = range(5, 51)
PER_HAC = 64
REPEATS = 3


def load_sample():
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    groups = {h: [] for h in HAC_RANGE}
    rows = [json.loads(line) for line in open(DATA) if line.strip()]
    random.seed(0)
    random.shuffle(rows)
    for r in rows:
        m = Chem.MolFromSmiles(r["smiles"])
        if m is None:
            continue
        h = m.GetNumHeavyAtoms()
        if h in groups and len(groups[h]) < PER_HAC:
            groups[h].append(r["smiles"])
    return {h: s for h, s in groups.items() if s}


def run_cpu():
    from nispo import smiles_to_iupac_batch
    from openclatura import name_smiles

    groups = load_sample()
    tools = {
        "openclatura": lambda s: name_smiles(s),
        "nispo": lambda s: smiles_to_iupac_batch([s])[0],
    }
    # warm up lazy initialisation on molecules of every size before timing
    warm = [g[0] for g in groups.values()] + [g[-1] for g in groups.values()]
    for fn in tools.values():
        for s in warm:
            try:
                fn(s)
            except Exception:
                pass
    # Tools are interleaved molecule by molecule so both see the same background
    # load, and each molecule is timed REPEATS times keeping the fastest run:
    # interference on a shared machine only ever adds time.
    out = {tool: {h: [] for h in groups} for tool in tools}
    for h, smiles in groups.items():
        for s in smiles:
            for tool, fn in tools.items():
                best = float("inf")
                for _ in range(REPEATS):
                    t = time.perf_counter()
                    try:
                        fn(s)
                    except Exception:
                        pass
                    best = min(best, time.perf_counter() - t)
                out[tool][h].append(best)
    print("cpu timing done")
    _save("cpu", out)


def run_gpu():
    import sys

    sys.path.insert(0, os.path.join(HERE, "..", "STOUT-pypi-2.0.5"))
    from STOUT import get_device_info, translate_forward_batch

    print(f"device: {get_device_info()}")
    groups = load_sample()
    translate_forward_batch(next(iter(groups.values())), batch_size=PER_HAC)  # warm up
    out = {"stout": {}}
    for h, smiles in groups.items():
        best = float("inf")
        for _ in range(2):
            t = time.perf_counter()
            translate_forward_batch(smiles, batch_size=len(smiles))
            best = min(best, time.perf_counter() - t)
        out["stout"][h] = {"batch_size": len(smiles), "per_molecule": best / len(smiles)}
        print(f"  hac {h}: n={len(smiles)} {best / len(smiles) * 1e3:.1f} ms/mol")
    _save("gpu", out)


def _save(part, data):
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, f"{part}.json"), "w") as f:
        json.dump(data, f)


def plot():
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import numpy as np

    import lama_aesthetics

    cpu = json.load(open(os.path.join(OUT_DIR, "cpu.json")))
    gpu = json.load(open(os.path.join(OUT_DIR, "gpu.json")))
    series = [
        ("Openclatura v0.4.0 (1 CPU core)", {int(h): np.mean(t) for h, t in cpu["openclatura"].items()}),
        ("NISPO v0.1.7 (1 CPU core)", {int(h): np.mean(t) for h, t in cpu["nispo"].items()}),
        ("STOUT v2.0.5 (H100, batched)", {int(h): v["per_molecule"] for h, v in gpu["stout"].items()}),
    ]

    lama_aesthetics.get_style("main")
    mpl.rcParams.update({
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    })
    fig, ax = plt.subplots(figsize=(lama_aesthetics.ONE_COL_WIDTH, lama_aesthetics.ONE_COL_HEIGHT))
    xs, ys = [], []
    for label, d in series:
        h = np.array(sorted(d))
        y = np.array([d[k] for k in h]) * 1e3
        ax.plot(h, y, marker="o", markersize=2, linewidth=1.0, label=label)
        xs.append(h)
        ys.append(y)
        print(f"{label}: {y[0]:.1f} ms at {h[0]} heavy atoms, {y[-1]:.1f} ms at {h[-1]}, "
              f"mean over counts {y.mean():.1f} ms")
    ax.set_yscale("log")
    ax.set_xlabel("Heavy atom count")
    ax.set_ylabel("Time per molecule (ms)")
    ax.set_ylim(0.5, 1000)
    ax.set_xlim(min(map(min, xs)) - 1, max(map(max, xs)) + 1)
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=1)
    out = os.path.join(HERE, "time_vs_size_batched")
    fig.savefig(out + ".png", dpi=300, bbox_inches="tight", pad_inches=0.1)
    fig.savefig(out + ".pdf", bbox_inches="tight", pad_inches=0.01)
    print(f"saved {out}.png / .pdf")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--part", choices=("cpu", "gpu", "plot"), required=True)
    {"cpu": run_cpu, "gpu": run_gpu, "plot": plot}[p.parse_args().part]()
