#!/usr/bin/env python3
"""Fast batched GPU STOUT runner using the modified STOUT-pypi-2.0.5 package.

Reads a JSONL of ``{"index", "smiles", ...}`` rows and writes a parallel JSONL
adding ``stout_iupac``, using ``STOUT.translate_forward_batch`` (batched greedy
decode on GPU). Output is identical to the stock single-item translate_forward;
only throughput differs.

Point CUDA_VISIBLE_DEVICES at the GPU you want (defaults to 0).

    python stout_batch_gpu.py --input data/qm9/qm9_all_input.jsonl \
        --output results/parity/qm9_full_pypi_batch.jsonl --batch-size 128
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Use the local modified STOUT-pypi-2.0.5 (batched) in preference to any
# pip-installed STOUT of the same import name.
sys.path.insert(0, str(Path(__file__).resolve().parent / "STOUT-pypi-2.0.5"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--limit", type=int, default=0, help="0 = all rows")
    parser.add_argument("--smiles-key", default="smiles")
    args = parser.parse_args(argv)

    rows = []
    with args.input.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if args.limit and len(rows) >= args.limit:
                break

    from STOUT import get_device_info, translate_forward_batch

    print(f"device: {get_device_info()}", file=sys.stderr)
    smiles = [r.get(args.smiles_key, "") for r in rows]

    start = time.perf_counter()
    names = translate_forward_batch(smiles, batch_size=args.batch_size)
    elapsed = time.perf_counter() - start

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as out:
        for i, (row, name) in enumerate(zip(rows, names)):
            out.write(
                json.dumps(
                    {"index": row.get("index", i), "smiles": row.get(args.smiles_key, ""), "stout_iupac": name}
                )
                + "\n"
            )

    rate = len(rows) / elapsed if elapsed else 0.0
    print(f"{len(rows)} molecules in {elapsed:.1f}s = {rate:.0f}/s -> {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
