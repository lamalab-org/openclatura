#!/usr/bin/env python3
"""STOUT-vs-OpenClatura evaluation driver.

Given a JSONL file of ``{"index": ..., "smiles": ...}`` rows (see
``evaluations/data/``), this script:

1. names every molecule with one or both backends (``openclatura`` and
   ``stout``), writing a JSONL file per backend that preserves the input
   fields and adds the IUPAC name (same layout as the existing
   ``*_openblue.jsonl`` / ``*_stout.jsonl`` result files);
2. round-trips each generated name back to SMILES through OPSIN and compares
   the canonical structure with the input, producing a failures CSV and a
   summary JSON per backend (same layout as the existing
   ``*_opsin_summary.json`` files); and
3. when both backends are run, prints a side-by-side accuracy comparison.

Everything runs inside the ``stout-pypi-eval`` conda environment, which holds
both ``openclatura`` and ``STOUT-pypi``. Nothing here is executed at import
time, so ``--help`` is cheap and does not load TensorFlow.

Examples
--------
Name + validate a 32-molecule QM9 smoke set with both backends::

    python evaluate.py --input data/qm9/qm9_all_input.jsonl \
        --outdir results/qm9 --backend both --limit 32

Reproduce a single PubChem STOUT shard::

    python evaluate.py --input data/pubchem/pubchem_seed42_100000_input.jsonl \
        --outdir results/pubchem --backend stout

Reproduce a legacy OpenBlue shard (keeps the ``openblue_iupac`` key)::

    python evaluate.py --input data/pubchem/pubchem_seed42_100000_input.jsonl \
        --outdir results/pubchem --backend openclatura
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from backends import BACKENDS, DEFAULT_NAME_KEY, load_backend


# --------------------------------------------------------------------------- #
# IO helpers
# --------------------------------------------------------------------------- #
def read_rows(path: Path, limit: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


# --------------------------------------------------------------------------- #
# Naming
# --------------------------------------------------------------------------- #
def name_rows(
    rows: list[dict[str, Any]],
    backend: str,
    name_key: str,
    smiles_key: str,
    progress_every: int,
) -> list[dict[str, Any]]:
    """Add ``name_key`` to every row using ``backend``; preserve input fields."""
    namer = load_backend(backend)
    total = len(rows)
    start = time.perf_counter()
    out: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        record = dict(row)
        record.setdefault("index", i)
        record[name_key] = namer(record.get(smiles_key, ""))
        out.append(record)
        if progress_every and (i + 1) % progress_every == 0:
            rate = (i + 1) / (time.perf_counter() - start)
            print(f"  [{backend}] {i + 1}/{total} ({rate:.0f}/s)", file=sys.stderr)
    elapsed = time.perf_counter() - start
    rate = total / elapsed if elapsed else 0.0
    print(f"[{backend}] named {total} molecules in {elapsed:.1f}s ({rate:.0f}/s)", file=sys.stderr)
    return out


# --------------------------------------------------------------------------- #
# OPSIN round-trip validation
# --------------------------------------------------------------------------- #
def _canonical(smiles: str) -> str | None:
    from rdkit import Chem

    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def _opsin_roundtrip(names: list[str]) -> list[str]:
    from py2opsin import py2opsin

    tmp_dir = Path(tempfile.gettempdir()) / "py2opsin_evaluations"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_dir / f"opsin_{os.getpid()}_{time.time_ns()}.txt"
    results = list(py2opsin(names, tmp_fpath=str(tmp_path)))
    if len(results) != len(names):
        raise RuntimeError(f"OPSIN returned {len(results)} rows for {len(names)} inputs")
    return results


def validate(
    rows: list[dict[str, Any]],
    name_key: str,
    smiles_key: str,
    failures_csv: Path,
) -> dict[str, Any]:
    """Round-trip names through OPSIN; write failures CSV; return summary dict."""
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")

    names = [str(row.get(name_key) or "") for row in rows]
    start = time.perf_counter()
    reconstructed = _opsin_roundtrip(names)

    counts = {
        "rows": len(rows),
        "original_valid": 0,
        "opsin_nonempty": 0,
        "opsin_valid": 0,
        "matches": 0,
        "failures": 0,
    }
    failures: list[dict[str, Any]] = []

    for row, opsin_smiles in zip(rows, reconstructed):
        original = row.get(smiles_key)
        name = row.get(name_key)
        original_canon = _canonical(original or "")
        if original_canon is not None:
            counts["original_valid"] += 1

        opsin_smiles = opsin_smiles or ""
        if opsin_smiles:
            counts["opsin_nonempty"] += 1
        opsin_canon = _canonical(opsin_smiles)
        if opsin_canon is not None:
            counts["opsin_valid"] += 1

        matched = (
            original_canon is not None
            and opsin_canon is not None
            and original_canon == opsin_canon
        )
        if matched:
            counts["matches"] += 1
        else:
            counts["failures"] += 1
            failures.append(
                {
                    "index": row.get("index"),
                    "smiles": original,
                    "name": name,
                    "opsin_smiles": opsin_smiles,
                    "original_canonical": original_canon or "",
                    "opsin_canonical": opsin_canon or "",
                }
            )

    elapsed = time.perf_counter() - start

    failures_csv.parent.mkdir(parents=True, exist_ok=True)
    with failures_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["index", "smiles", "name", "opsin_smiles", "original_canonical", "opsin_canonical"],
        )
        writer.writeheader()
        writer.writerows(failures)

    accuracy = 100.0 * counts["matches"] / counts["rows"] if counts["rows"] else 0.0
    summary = {
        **counts,
        "accuracy": round(accuracy, 3),
        "elapsed_seconds": elapsed,
        "rows_per_second": counts["rows"] / elapsed if elapsed else 0.0,
        "failures_csv": str(failures_csv),
    }
    return summary


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run_backend(
    rows: list[dict[str, Any]],
    backend: str,
    outdir: Path,
    stem: str,
    name_key: str,
    smiles_key: str,
    validate_names: bool,
    progress_every: int,
) -> dict[str, Any]:
    named = name_rows(rows, backend, name_key, smiles_key, progress_every)
    names_path = outdir / f"{stem}_{backend}.jsonl"
    write_jsonl(names_path, named)
    print(f"[{backend}] wrote {names_path}", file=sys.stderr)

    result: dict[str, Any] = {"backend": backend, "names_jsonl": str(names_path)}
    if validate_names:
        failures_csv = outdir / f"{stem}_{backend}_opsin_failures.csv"
        summary = validate(named, name_key, smiles_key, failures_csv)
        summary_path = outdir / f"{stem}_{backend}_opsin_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(
            f"[{backend}] accuracy {summary['accuracy']}% "
            f"({summary['matches']}/{summary['rows']}) -> {summary_path}",
            file=sys.stderr,
        )
        result["summary"] = summary
        result["summary_json"] = str(summary_path)
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, type=Path, help="input JSONL of {index, smiles} rows")
    parser.add_argument("--outdir", required=True, type=Path, help="directory for name/validation outputs")
    parser.add_argument(
        "--backend",
        choices=(*BACKENDS, "both"),
        default="both",
        help="which namer(s) to run (default: both)",
    )
    parser.add_argument("--smiles-key", default="smiles", help="JSONL key holding the input SMILES")
    parser.add_argument(
        "--name-key",
        default=None,
        help="JSONL key to store the generated name; default per backend "
        "(openclatura->openblue_iupac, stout->stout_iupac)",
    )
    parser.add_argument("--limit", type=int, default=0, help="only process the first N rows (0 = all)")
    parser.add_argument("--no-validate", action="store_true", help="skip the OPSIN round-trip validation")
    parser.add_argument("--progress-every", type=int, default=10000, help="progress log cadence in molecules")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rows = read_rows(args.input, args.limit)
    if not rows:
        print(f"no rows read from {args.input}", file=sys.stderr)
        return 1

    stem = args.input.stem.replace("_input", "")
    backends = list(BACKENDS) if args.backend == "both" else [args.backend]
    args.outdir.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict[str, Any]] = {}
    for backend in backends:
        name_key = args.name_key or DEFAULT_NAME_KEY[backend]
        results[backend] = run_backend(
            rows,
            backend,
            args.outdir,
            stem,
            name_key,
            args.smiles_key,
            validate_names=not args.no_validate,
            progress_every=args.progress_every,
        )

    if not args.no_validate and len(backends) > 1:
        print("\n=== STOUT vs OpenClatura ===")
        header = f"{'backend':<14}{'accuracy':>10}{'matches':>12}{'rows':>10}"
        print(header)
        print("-" * len(header))
        for backend in backends:
            s = results[backend].get("summary", {})
            print(f"{backend:<14}{s.get('accuracy', 0):>9}%{s.get('matches', 0):>12}{s.get('rows', 0):>10}")

    comparison_path = args.outdir / f"{stem}_comparison.json"
    comparison_path.write_text(
        json.dumps({b: results[b].get("summary", {}) for b in backends}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"\nwrote comparison summary -> {comparison_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
