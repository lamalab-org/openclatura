#!/usr/bin/env python3
"""Compare single-thread naming speed for two OpenClatura source trees."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def _load_smiles(corpus: Path) -> list[str]:
    with corpus.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    smiles = [row["smiles"] for row in rows if row.get("smiles")]
    if not smiles:
        raise ValueError(f"no SMILES found in {corpus}")
    return smiles


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def measure(corpus: Path, warmup_rows: int) -> dict[str, Any]:
    from rdkit import RDLogger

    from openclatura import __version__, name_many

    RDLogger.DisableLog("rdApp.*")
    smiles = _load_smiles(corpus)
    warmup = smiles[: min(warmup_rows, len(smiles))]
    warmup_results = name_many(warmup, processes=1)
    if any(not result.name for result in warmup_results):
        raise RuntimeError("one or more warm-up molecules could not be named")

    started = time.perf_counter()
    results = name_many(smiles, processes=1)
    elapsed = time.perf_counter() - started
    names = [result.name or "" for result in results]
    if len(names) != len(smiles) or any(not name for name in names):
        named = sum(bool(name) for name in names)
        raise RuntimeError(f"only {named}/{len(smiles)} benchmark molecules were named")

    checksum = hashlib.sha256("\n".join(names).encode()).hexdigest()
    return {
        "version": __version__,
        "rows": len(smiles),
        "elapsed_seconds": elapsed,
        "rows_per_second": len(smiles) / elapsed,
        "names_sha256": checksum,
    }


def _run_measurement(source_tree: Path, corpus: Path, warmup_rows: int) -> dict[str, Any]:
    env = os.environ.copy()
    env.update(
        {
            "PYTHONHASHSEED": "0",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "PYTHONPATH": str(source_tree.resolve()),
        }
    )
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--measure",
        "--corpus",
        str(corpus.resolve()),
        "--warmup-rows",
        str(warmup_rows),
    ]
    # `command` is an argv list and shell=False remains in effect, so path
    # characters cannot be interpreted as shell syntax.
    completed = subprocess.run(  # nosemgrep: python.lang.security.audit.dangerous-subprocess-use-audit
        command, env=env, check=False, capture_output=True, text=True
    )
    if completed.returncode:
        raise RuntimeError(f"measurement failed for {source_tree}:\n{completed.stdout}\n{completed.stderr}".rstrip())
    return json.loads(completed.stdout)


def _summary_markdown(report: dict[str, Any]) -> str:
    status = "PASS" if report["passed"] else "FAIL"
    lines = [
        "## Single-thread performance benchmark",
        "",
        f"**{status}** — paired median change: **{report['regression_percent']:+.2f}%**",
        "",
        "| Revision | Median seconds | Median molecules/s |",
        "| --- | ---: | ---: |",
        f"| Base | {report['base_median_seconds']:.3f} | {report['base_median_rows_per_second']:.1f} |",
        f"| PR | {report['head_median_seconds']:.3f} | {report['head_median_rows_per_second']:.1f} |",
        "",
        f"The gate fails only above {report['threshold_percent']:.1f}% and "
        f"{report['minimum_absolute_seconds']:.1f}s slower.",
    ]
    return "\n".join(lines) + "\n"


def compare(args: argparse.Namespace) -> int:
    measurements: dict[str, list[dict[str, Any]]] = {"base": [], "head": []}
    paired_ratios: list[float] = []
    paired_deltas: list[float] = []

    # Reverse every other pair to reduce bias from runner warm-up or drift.
    for repetition in range(args.repetitions):
        order = ("base", "head") if repetition % 2 == 0 else ("head", "base")
        current: dict[str, dict[str, Any]] = {}
        for label in order:
            source = args.base_src if label == "base" else args.head_src
            print(f"measurement {repetition + 1}/{args.repetitions}: {label}", file=sys.stderr)
            result = _run_measurement(source, args.corpus, args.warmup_rows)
            measurements[label].append(result)
            current[label] = result
            print(
                f"  {result['elapsed_seconds']:.3f}s ({result['rows_per_second']:.1f} molecules/s)",
                file=sys.stderr,
            )
        base_seconds = current["base"]["elapsed_seconds"]
        head_seconds = current["head"]["elapsed_seconds"]
        paired_ratios.append(head_seconds / base_seconds)
        paired_deltas.append(head_seconds - base_seconds)

    base_times = [row["elapsed_seconds"] for row in measurements["base"]]
    head_times = [row["elapsed_seconds"] for row in measurements["head"]]
    paired_ratio = statistics.median(paired_ratios)
    paired_delta = statistics.median(paired_deltas)
    regression_percent = (paired_ratio - 1.0) * 100.0
    passed = not (regression_percent > args.threshold_percent and paired_delta > args.minimum_absolute_seconds)
    rows = measurements["head"][0]["rows"]
    report = {
        "passed": passed,
        "rows": rows,
        "repetitions": args.repetitions,
        "threshold_percent": args.threshold_percent,
        "minimum_absolute_seconds": args.minimum_absolute_seconds,
        "regression_percent": regression_percent,
        "paired_median_delta_seconds": paired_delta,
        "base_median_seconds": statistics.median(base_times),
        "head_median_seconds": statistics.median(head_times),
        "base_median_rows_per_second": rows / statistics.median(base_times),
        "head_median_rows_per_second": rows / statistics.median(head_times),
        "paired_ratios": paired_ratios,
        "measurements": measurements,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    summary = _summary_markdown(report)
    print(summary)
    if github_summary := os.environ.get("GITHUB_STEP_SUMMARY"):
        with Path(github_summary).open("a", encoding="utf-8") as handle:
            handle.write(summary)
    return 0 if passed else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measure", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--base-src", type=Path)
    parser.add_argument("--head-src", type=Path)
    parser.add_argument("--repetitions", type=_positive_int, default=3)
    parser.add_argument("--warmup-rows", type=int, default=100)
    parser.add_argument("--threshold-percent", type=float, default=15.0)
    parser.add_argument("--minimum-absolute-seconds", type=float, default=1.0)
    parser.add_argument("--report", type=Path, default=Path("performance-report.json"))
    args = parser.parse_args()
    if not args.measure and (args.base_src is None or args.head_src is None):
        parser.error("--base-src and --head-src are required for comparison")
    return args


def main() -> int:
    args = parse_args()
    if args.measure:
        print(json.dumps(measure(args.corpus, args.warmup_rows)))
        return 0
    return compare(args)


if __name__ == "__main__":
    raise SystemExit(main())
