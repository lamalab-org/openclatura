#!/usr/bin/env python3
"""Run ONE model on ONE input file and write ONE output jsonl.

Each output line is a single molecule's result: ``{"index", "smiles", <key>}``
where <key> is ``stout_iupac`` or ``openclatura_iupac``. One input file + one
model => one self-contained jsonl (no bundled comparison/scoring files), so
every jsonl corresponds to exactly one output.

Models
------
* ``stout``       - batched GPU decode via the local STOUT-pypi-2.0.5 package
                    (identical output to single-item CPU translate_forward).
                    Set CUDA_VISIBLE_DEVICES to pick the GPU.
* ``openclatura`` - our deterministic namer, multiprocessed across CPU cores.

Examples
--------
    CUDA_VISIBLE_DEVICES=1 python predict.py --model stout \
        --input data/pubchem/pubchem_seed42_100000_input.jsonl \
        --output results/pubchem/pubchem_seed42_100000_stout.jsonl --batch-size 128

    python predict.py --model openclatura \
        --input data/qm9/qm9_all_input.jsonl \
        --output results/qm9/qm9_all_openclatura.jsonl --workers 64
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

NAME_KEY = {"stout": "stout_iupac", "openclatura": "openclatura_iupac"}


def _read_rows(path: Path, limit: int) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def _write_rows(path: Path, rows: list[dict], key: str, names: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as out:
        for i, (row, name) in enumerate(zip(rows, names)):
            out.write(
                json.dumps({"index": row.get("index", i), "smiles": row.get("smiles", ""), key: name or ""})
                + "\n"
            )


# --------------------------------------------------------------------------- #
# STOUT (batched GPU)
# --------------------------------------------------------------------------- #
def _run_stout(smiles: list[str], batch_size: int) -> list[str]:
    sys.path.insert(0, str(Path(__file__).resolve().parent / "STOUT-pypi-2.0.5"))
    from STOUT import get_device_info, translate_forward_batch

    print(f"device: {get_device_info()}", file=sys.stderr)
    return translate_forward_batch(smiles, batch_size=batch_size)


# --------------------------------------------------------------------------- #
# openclatura (CPU, multiprocessed)
# --------------------------------------------------------------------------- #
_OC_NAME = None


def _oc_init():
    global _OC_NAME
    from openclatura import name_smiles

    _OC_NAME = name_smiles


def _oc_chunk(chunk: list[str]) -> list[str]:
    out = []
    for smi in chunk:
        if not smi:
            out.append("")
            continue
        try:
            out.append(_OC_NAME(smi) or "")
        except Exception:
            out.append("")
    return out


def _run_openclatura(smiles: list[str], workers: int, chunk: int) -> list[str]:
    if workers <= 1:
        _oc_init()
        return _oc_chunk(smiles)

    from concurrent.futures import ProcessPoolExecutor

    chunks = [smiles[i : i + chunk] for i in range(0, len(smiles), chunk)]
    results: list[str] = []
    with ProcessPoolExecutor(max_workers=workers, initializer=_oc_init) as ex:
        for part in ex.map(_oc_chunk, chunks):
            results.extend(part)
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=("stout", "openclatura"), required=True)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=0, help="0 = all rows")
    parser.add_argument("--batch-size", type=int, default=128, help="stout: GPU batch size (<=256)")
    parser.add_argument("--workers", type=int, default=min(64, os.cpu_count() or 1), help="openclatura CPU workers")
    parser.add_argument("--chunk", type=int, default=1000, help="openclatura molecules per worker task")
    args = parser.parse_args(argv)

    rows = _read_rows(args.input, args.limit)
    if not rows:
        print(f"no rows in {args.input}", file=sys.stderr)
        return 1
    smiles = [r.get("smiles", "") for r in rows]

    start = time.perf_counter()
    if args.model == "stout":
        names = _run_stout(smiles, args.batch_size)
    else:
        names = _run_openclatura(smiles, args.workers, args.chunk)
    elapsed = time.perf_counter() - start

    _write_rows(args.output, rows, NAME_KEY[args.model], names)
    rate = len(rows) / elapsed if elapsed else 0.0
    print(
        f"[{args.model}] {len(rows)} molecules in {elapsed:.1f}s = {rate:.0f}/s -> {args.output}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
