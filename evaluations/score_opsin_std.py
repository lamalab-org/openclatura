#!/usr/bin/env python3
"""Score prediction jsonls by OPSIN round-trip with full structure standardization.

Match rule: parse the generated IUPAC name back to SMILES with OPSIN, then run
BOTH the input SMILES and the OPSIN SMILES through
``standardize_and_canonicalize_tautomer`` (RDKit Cleanup -> normalize -> reionize
-> uncharge -> tautomer canonicalize) and compare. Counts a molecule correct even
when OPSIN returns a different-but-equivalent tautomer/charge/normalization form.

Fully multicore: the entire per-molecule pipeline -- OPSIN round-trip (its own
JVM per chunk) AND standardization -- runs inside one shared process pool, so
every core is busy the whole time (no serial OPSIN phase). Writes/overwrites
``<stem>_opsin_summary.json`` + ``<stem>_opsin_failures.csv`` next to each input.

    python score_opsin_std.py results/*/*_stout.jsonl --name-key stout_iupac --workers 64
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import evaluate as E  # read_rows

_W: dict = {}


def _init_worker():
    from py2opsin import py2opsin
    from rdkit import Chem, RDLogger
    from rdkit.Chem.MolStandardize import rdMolStandardize

    RDLogger.DisableLog("rdApp.*")
    _W.update(
        py2opsin=py2opsin,
        Chem=Chem,
        rms=rdMolStandardize,
        normalizer=rdMolStandardize.Normalizer(),
        reionizer=rdMolStandardize.Reionizer(),
        uncharger=rdMolStandardize.Uncharger(),
        tautomer=rdMolStandardize.TautomerEnumerator(),
    )


def standardize_and_canonicalize_tautomer(smi):
    Chem = _W["Chem"]
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    try:
        mol = _W["rms"].Cleanup(mol)
        mol = _W["normalizer"].normalize(mol)
        mol = _W["reionizer"].reionize(mol)
        mol = _W["uncharger"].uncharge(mol)
        mol = _W["tautomer"].Canonicalize(mol)
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def _process_chunk(payload):
    """OPSIN round-trip + standardize + compare for one chunk of rows."""
    rows, name_key, smiles_key = payload
    names = [str(r.get(name_key) or "") for r in rows]

    tmp_dir = Path(tempfile.gettempdir()) / "opsin_std"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_dir / f"opsin_{os.getpid()}_{time.time_ns()}.txt"
    opsin_smiles = list(_W["py2opsin"](names, tmp_fpath=str(tmp_path)))
    if len(opsin_smiles) != len(names):  # defensive: keep alignment
        opsin_smiles = [_W["py2opsin"](n, tmp_fpath=str(tmp_path)) if n else "" for n in names]

    counts = {"rows": 0, "original_valid": 0, "opsin_nonempty": 0, "opsin_valid": 0, "matches": 0, "failures": 0}
    failures = []
    for row, name, opsin in zip(rows, names, opsin_smiles):
        counts["rows"] += 1
        orig = row.get(smiles_key, "")
        opsin = opsin or ""
        std_orig = standardize_and_canonicalize_tautomer(orig) if orig else None
        std_opsin = standardize_and_canonicalize_tautomer(opsin) if opsin else None
        if std_orig is not None:
            counts["original_valid"] += 1
        if opsin:
            counts["opsin_nonempty"] += 1
        if std_opsin is not None:
            counts["opsin_valid"] += 1
        if std_orig is not None and std_opsin is not None and std_orig == std_opsin:
            counts["matches"] += 1
        else:
            counts["failures"] += 1
            failures.append(
                {
                    "index": row.get("index"),
                    "smiles": orig,
                    "name": name,
                    "opsin_smiles": opsin,
                    "std_original": std_orig or "",
                    "std_opsin": std_opsin or "",
                }
            )
    return counts, failures


def _score_file(path: Path, name_key: str, smiles_key: str, chunk: int, pool: ProcessPoolExecutor) -> dict:
    rows = E.read_rows(path)
    payloads = [(rows[i : i + chunk], name_key, smiles_key) for i in range(0, len(rows), chunk)]

    counts = {"rows": 0, "original_valid": 0, "opsin_nonempty": 0, "opsin_valid": 0, "matches": 0, "failures": 0}
    failures_csv = path.with_name(f"{path.stem}_opsin_failures.csv")
    with failures_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["index", "smiles", "name", "opsin_smiles", "std_original", "std_opsin"])
        writer.writeheader()
        for c, fails in pool.map(_process_chunk, payloads):
            for k in counts:
                counts[k] += c[k]
            writer.writerows(fails)

    accuracy = 100.0 * counts["matches"] / counts["rows"] if counts["rows"] else 0.0
    summary = {
        **counts,
        "accuracy": round(accuracy, 3),
        "match_method": "standardize_and_canonicalize_tautomer",
        "failures_csv": str(failures_csv),
    }
    (path.with_name(f"{path.stem}_opsin_summary.json")).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--name-key", required=True)
    parser.add_argument("--smiles-key", default="smiles")
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--chunk", type=int, default=1000)
    args = parser.parse_args(argv)

    rows_total = matches_total = 0
    table = []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker) as pool:
        for path in args.inputs:
            if not path.exists():
                print(f"skip missing {path}", file=sys.stderr)
                continue
            t = time.perf_counter()
            s = _score_file(path, args.name_key, args.smiles_key, args.chunk, pool)
            dt = time.perf_counter() - t
            rows_total += s["rows"]
            matches_total += s["matches"]
            table.append((path.name, s["accuracy"], s["matches"], s["rows"]))
            print(f"  {path.name:<45} {s['accuracy']:>7}%  {s['matches']}/{s['rows']}  ({dt:.0f}s)", file=sys.stderr)

    print("\n=== OPSIN round-trip accuracy (standardized + tautomer-canonical) ===")
    print(f"{'file':<45}{'acc':>9}{'matches':>12}{'rows':>10}")
    print("-" * 76)
    for name, acc, m, n in table:
        print(f"{name:<45}{acc:>8}%{m:>12}{n:>10}")
    agg = 100.0 * matches_total / rows_total if rows_total else 0.0
    print("-" * 76)
    print(f"{'AGGREGATE':<45}{round(agg, 3):>8}%{matches_total:>12}{rows_total:>10}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
