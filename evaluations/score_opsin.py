#!/usr/bin/env python3
"""Score prediction jsonls by OPSIN round-trip accuracy.

For each ``*_<model>.jsonl`` produced by predict.py, parse the generated IUPAC
name back to SMILES with OPSIN and compare (canonical RDKit SMILES) to the input
SMILES. Writes ``<stem>_opsin_summary.json`` + ``<stem>_opsin_failures.csv`` next
to each input and prints a per-file + aggregate accuracy table.

    python score_opsin.py results/*/*_openclatura.jsonl --name-key openclatura_iupac
    python score_opsin.py results/qm9/qm9_all_stout.jsonl --name-key stout_iupac
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import evaluate as E  # reuse read_rows + validate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="+", type=Path, help="prediction jsonl files")
    parser.add_argument("--name-key", required=True, help="jsonl key holding the IUPAC name")
    parser.add_argument("--smiles-key", default="smiles")
    args = parser.parse_args(argv)

    rows_total = matches_total = 0
    table = []
    for path in args.inputs:
        if not path.exists():
            print(f"skip missing {path}", file=sys.stderr)
            continue
        rows = E.read_rows(path)
        failures_csv = path.with_name(f"{path.stem}_opsin_failures.csv")
        summary = E.validate(rows, args.name_key, args.smiles_key, failures_csv)
        summary_path = path.with_name(f"{path.stem}_opsin_summary.json")
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        rows_total += summary["rows"]
        matches_total += summary["matches"]
        table.append((path.name, summary["accuracy"], summary["matches"], summary["rows"]))
        print(f"  {path.name:<45} {summary['accuracy']:>7}%  {summary['matches']}/{summary['rows']}", file=sys.stderr)

    print("\n=== OPSIN round-trip accuracy ===")
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
