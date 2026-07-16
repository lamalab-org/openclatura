#!/usr/bin/env python3
"""Compatibility check for the local STOUT-pypi-2.0.5 batched-GPU package.

Confirms that ``translate_forward_batch`` (batched greedy decode on GPU) returns
exactly the same IUPAC names as the stock single-item ``translate_forward`` run
on CPU. Both come from the same local package, ``evaluations/STOUT-pypi-2.0.5``
-- the only STOUT used for comparison now that the modified fork was removed.

Because the batched path reuses the SavedModel's own trained transformer and the
identical argmax greedy rule, the two are expected to match token-for-token;
this script quantifies that on a sample (verified 50/50 on random PubChem).

The two backends run as separate subprocesses so the CPU reference and the GPU
batch each get their own ``CUDA_VISIBLE_DEVICES``.

Usage
-----
    python stout_parity.py compare --input data/pubchem/pubchem_seed42_100000_input.jsonl \
        --outdir results/parity --limit 50 --gpu 1

    # single backend (used internally by compare):
    python stout_parity.py run --mode single \
        --input data/qm9/qm9_all_input.jsonl --output out_single.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

LOCAL_STOUT = Path(__file__).resolve().parent / "STOUT-pypi-2.0.5"


# --------------------------------------------------------------------------- #
# run mode: emit names with the local package, single-item or batched
# --------------------------------------------------------------------------- #
def run_mode(args: argparse.Namespace) -> int:
    sys.path.insert(0, str(LOCAL_STOUT))
    import STOUT

    rows: list[dict[str, Any]] = []
    with open(args.input, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if args.limit and len(rows) >= args.limit:
                break

    smiles = [r.get("smiles", "") for r in rows]
    start = time.perf_counter()
    if args.mode == "single":
        names = [STOUT.translate_forward(s) if s else "" for s in smiles]
    else:
        names = STOUT.translate_forward_batch(smiles, batch_size=args.batch_size)
    elapsed = time.perf_counter() - start

    with open(args.output, "w", encoding="utf-8") as out:
        for i, (row, name) in enumerate(zip(rows, names)):
            out.write(
                json.dumps(
                    {"index": row.get("index", i), "smiles": row.get("smiles", ""), "stout_iupac": name or ""}
                )
                + "\n"
            )
    print(f"[{args.mode}] {len(rows)} molecules in {elapsed:.1f}s -> {args.output}", file=sys.stderr)
    return 0


# --------------------------------------------------------------------------- #
# compare mode: CPU single reference vs GPU batched
# --------------------------------------------------------------------------- #
def _run_subprocess(mode: str, cuda_devices: str, input_path: Path, output_path: Path, limit: int) -> None:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "run",
        "--mode",
        mode,
        "--input",
        str(input_path),
        "--output",
        str(output_path),
    ]
    if limit:
        cmd += ["--limit", str(limit)]
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = cuda_devices
    where = "CPU" if cuda_devices == "" else f"GPU {cuda_devices}"
    print(f"[compare] {mode} on {where}", file=sys.stderr)
    subprocess.run(cmd, check=True, env=env)


def _read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def compare_mode(args: argparse.Namespace) -> int:
    args.outdir.mkdir(parents=True, exist_ok=True)
    single_out = args.outdir / "parity_cpu_single.jsonl"
    batch_out = args.outdir / "parity_gpu_batch.jsonl"

    _run_subprocess("single", "", args.input, single_out, args.limit)
    _run_subprocess("batch", str(args.gpu), args.input, batch_out, args.limit)

    single_rows = _read(single_out)
    batch_rows = _read(batch_out)
    n = min(len(single_rows), len(batch_rows))

    diffs = []
    for i in range(n):
        a = single_rows[i].get("stout_iupac", "")
        b = batch_rows[i].get("stout_iupac", "")
        if a != b:
            diffs.append(
                {
                    "index": single_rows[i].get("index", i),
                    "smiles": single_rows[i].get("smiles", ""),
                    "cpu_single_iupac": a,
                    "gpu_batch_iupac": b,
                }
            )

    diff_path = args.outdir / "parity_diffs.jsonl"
    with diff_path.open("w", encoding="utf-8") as handle:
        for d in diffs:
            handle.write(json.dumps(d) + "\n")

    identical = n - len(diffs)
    summary = {
        "compared": n,
        "identical": identical,
        "different": len(diffs),
        "identical_pct": round(100.0 * identical / n, 3) if n else 0.0,
        "cpu_single_jsonl": str(single_out),
        "gpu_batch_jsonl": str(batch_out),
        "diffs_jsonl": str(diff_path),
    }
    (args.outdir / "parity_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print("\n=== STOUT-pypi-2.0.5 compatibility: CPU single vs GPU batched ===")
    print(f"compared:  {n}")
    print(f"identical: {identical} ({summary['identical_pct']}%)")
    print(f"different: {len(diffs)}")
    for d in diffs[:5]:
        print(f"  [{d['index']}] {d['smiles']}")
        print(f"      cpu-single: {d['cpu_single_iupac']}")
        print(f"      gpu-batch : {d['gpu_batch_iupac']}")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode_cmd", required=True)

    run_p = sub.add_parser("run", help="emit names with the local package")
    run_p.add_argument("--mode", choices=("single", "batch"), required=True)
    run_p.add_argument("--input", required=True)
    run_p.add_argument("--output", required=True)
    run_p.add_argument("--limit", type=int, default=0)
    run_p.add_argument("--batch-size", type=int, default=64)

    cmp_p = sub.add_parser("compare", help="CPU single vs GPU batched on a sample")
    cmp_p.add_argument("--input", type=Path, required=True)
    cmp_p.add_argument("--outdir", type=Path, required=True)
    cmp_p.add_argument("--limit", type=int, default=50)
    cmp_p.add_argument("--gpu", default="1", help="GPU id for the batched run (default 1)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.mode_cmd == "run":
        return run_mode(args)
    return compare_mode(args)


if __name__ == "__main__":
    raise SystemExit(main())
