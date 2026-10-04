#!/usr/bin/env python3
"""Rerun a saved Openclatura failure corpus and score it with the paper metric."""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import signal
import tempfile
import time
import warnings
from collections import Counter
from pathlib import Path

_TIMEOUT_SECONDS = 30


def _init_namer(timeout_seconds: int) -> None:
    global _TIMEOUT_SECONDS
    _TIMEOUT_SECONDS = timeout_seconds
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")


def _name_one(payload: tuple[int, dict[str, str]]) -> tuple[int, dict[str, str]]:
    position, row = payload

    def _timeout(_signum, _frame):
        raise TimeoutError("per-molecule timeout")

    signal.signal(signal.SIGALRM, _timeout)
    signal.alarm(_TIMEOUT_SECONDS)
    try:
        from openclatura import name

        result = name(row["smiles"])
        if result.error:
            status = "error"
            error = result.error
        elif not result.name:
            status = "abstained"
            error = ""
        else:
            status = "ok"
            error = ""
        updated = {
            **row,
            "name": result.name or "",
            "naming_status": status,
            "naming_error": error,
        }
    except Exception as exc:  # noqa: BLE001 - every failure is evaluation data
        updated = {
            **row,
            "name": "",
            "naming_status": "error",
            "naming_error": f"{type(exc).__name__}: {exc}",
        }
    finally:
        signal.alarm(0)
    return position, updated


def _inchi_key(smiles: str) -> str:
    if not smiles:
        return ""
    try:
        from rdkit import Chem, RDLogger
        from rdkit.Chem import inchi

        RDLogger.DisableLog("rdApp.*")
        mol = Chem.MolFromSmiles(smiles)
        return inchi.MolToInchiKey(mol) if mol is not None else ""
    except Exception:  # noqa: BLE001 - invalid structures are an evaluation outcome
        return ""


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fields = [
        "index",
        "smiles",
        "name",
        "naming_status",
        "naming_error",
        "opsin_smiles",
        "outcome",
        "comparison_error",
        "previous_outcome",
        "previous_naming_error",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _opsin_decode(rows: list[dict[str, str]], chunk_size: int) -> None:
    from py2opsin import py2opsin

    emitted = [index for index, row in enumerate(rows) if row["naming_status"] == "ok" and row["name"]]
    for start in range(0, len(emitted), chunk_size):
        block = emitted[start : start + chunk_size]
        names = [rows[index]["name"] for index in block]
        tmp_path = Path(tempfile.gettempdir()) / f"openclatura_failure_opsin_{os.getpid()}_{start}.txt"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            decoded = list(
                py2opsin(
                    names,
                    allow_radicals=True,
                    tmp_fpath=str(tmp_path),
                )
            )
        tmp_path.unlink(missing_ok=True)
        if len(decoded) != len(block):
            raise RuntimeError(f"OPSIN alignment failure: got {len(decoded)} results for {len(block)} names")
        for index, smiles in zip(block, decoded):
            rows[index]["opsin_smiles"] = smiles or ""
        print(f"OPSIN {min(start + len(block), len(emitted))}/{len(emitted)}", flush=True)


def _score(rows: list[dict[str, str]], workers: int) -> Counter:
    with mp.Pool(workers) as pool:
        input_keys = pool.map(_inchi_key, [row["smiles"] for row in rows], chunksize=200)
        output_keys = pool.map(_inchi_key, [row.get("opsin_smiles", "") for row in rows], chunksize=200)

    counts: Counter = Counter()
    for row, input_key, output_key in zip(rows, input_keys, output_keys):
        row["comparison_error"] = ""
        if row["naming_status"] != "ok":
            outcome = "naming_error"
        elif not row.get("opsin_smiles"):
            outcome = "opsin_unparseable"
        elif not input_key or not output_key:
            outcome = "opsin_structure_invalid"
        elif input_key == output_key:
            outcome = "exact_match"
        elif input_key.split("-", 1)[0] == output_key.split("-", 1)[0]:
            outcome = "constitutional_match"
        else:
            outcome = "graph_mismatch"
        row["outcome"] = outcome
        counts[outcome] += 1
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=min(24, os.cpu_count() or 1))
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--opsin-chunk", type=int, default=5000)
    args = parser.parse_args()

    source_rows = _read_rows(args.input)
    rows = [
        {
            "index": row["index"],
            "smiles": row["smiles"],
            "previous_outcome": row.get("outcome", ""),
            "previous_naming_error": row.get("naming_error", ""),
        }
        for row in source_rows
    ]
    started = time.perf_counter()
    results: list[tuple[int, dict[str, str]]] = []
    with mp.Pool(args.workers, initializer=_init_namer, initargs=(args.timeout,)) as pool:
        for completed, result in enumerate(pool.imap_unordered(_name_one, enumerate(rows), chunksize=1), start=1):
            results.append(result)
            if completed % 5000 == 0:
                elapsed = time.perf_counter() - started
                print(f"Named {completed}/{len(rows)} ({completed / elapsed:.1f}/s)", flush=True)
    rows = [row for _, row in sorted(results)]

    _opsin_decode(rows, args.opsin_chunk)
    counts = _score(rows, args.workers)
    failures = [row for row in rows if row["outcome"] != "exact_match"]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_path = args.output_dir / "current_all.csv"
    failures_path = args.output_dir / "current_failures.csv"
    summary_path = args.output_dir / "current_summary.json"
    _write_csv(all_path, rows)
    _write_csv(failures_path, failures)
    summary = {
        "source": str(args.input),
        "rows": len(rows),
        "counts": dict(sorted(counts.items())),
        "exact_recovered_from_previous_failures": counts["exact_match"],
        "remaining_failures": len(failures),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "metric": "OPSIN 2.9.0 allow_radicals=True; full InChIKey exact match",
        "all_rows_csv": str(all_path),
        "failures_csv": str(failures_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
