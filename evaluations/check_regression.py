#!/usr/bin/env python3
"""Check OpenClatura accuracy against a precomputed evaluation shard.

Names equal to the stored result reuse the stored OPSIN pass/fail outcome.
Only changed names are round-tripped through OPSIN.  This keeps the check
faithful to the paper evaluation while avoiding more than a million redundant
OPSIN calls on an unchanged implementation.

The command exits non-zero when the current number of valid round trips is
lower than the stored baseline::

    python evaluations/check_regression.py \
        evaluations/results/qm9/qm9_all_openclatura.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

NAME_KEY = "openclatura_iupac"


def _companion_paths(baseline_path: Path) -> tuple[Path, Path]:
    stem = baseline_path.stem
    return (
        baseline_path.with_name(f"{stem}_opsin_summary.json"),
        baseline_path.with_name(f"{stem}_opsin_failures.csv"),
    )


def _row_id(row: dict[str, Any]) -> str:
    """Return the stable identifier used by the stored failure CSV."""

    return str(row["index"])


def load_baseline(baseline_path: Path, limit: int = 0) -> tuple[list[dict[str, Any]], set[str], dict[str, Any]]:
    """Load prediction rows, historical failure ids, and summary metadata."""

    summary_path, failures_path = _companion_paths(baseline_path)
    missing = [path for path in (baseline_path, summary_path, failures_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing baseline file(s): " + ", ".join(map(str, missing)))

    rows: list[dict[str, Any]] = []
    with baseline_path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if "index" not in row or "smiles" not in row or NAME_KEY not in row:
                    raise ValueError(f"invalid baseline row in {baseline_path}: {row!r}")
                rows.append(row)
                if limit and len(rows) >= limit:
                    break

    with failures_path.open(newline="", encoding="utf-8") as handle:
        failure_ids = {str(row["index"]) for row in csv.DictReader(handle)}

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not limit:
        if len(rows) != summary["rows"]:
            raise ValueError(f"baseline row count is {len(rows)}, summary says {summary['rows']}")
        if len(failure_ids) != summary["failures"]:
            raise ValueError(f"baseline failure CSV has {len(failure_ids)} rows, summary says {summary['failures']}")
        if len(rows) - len(failure_ids) != summary["matches"]:
            raise ValueError("baseline summary matches/failures are inconsistent")

    return rows, failure_ids, summary


def _opsin_roundtrip(names: list[str]) -> list[str]:
    try:
        from py2opsin import py2opsin
    except Exception as exc:  # pragma: no cover - depends on optional CI dependency
        raise RuntimeError("changed names require py2opsin") from exc

    tmp_path = Path(tempfile.gettempdir()) / f"openclatura_regression_{os.getpid()}_{time.time_ns()}.txt"
    try:
        decoded = list(py2opsin(names, tmp_fpath=str(tmp_path)))
    finally:
        tmp_path.unlink(missing_ok=True)
    if len(decoded) != len(names):
        raise RuntimeError(f"OPSIN returned {len(decoded)} rows for {len(names)} names")
    return [value or "" for value in decoded]


def verify_changed_rows(changed: list[dict[str, Any]], opsin_chunk_size: int) -> None:
    """Mutate discrepancy records with their current OPSIN result."""

    from openclatura.utils import standardize_mol

    nonempty = [row for row in changed if row["current_name"]]
    for start in range(0, len(nonempty), opsin_chunk_size):
        chunk = nonempty[start : start + opsin_chunk_size]
        decoded = _opsin_roundtrip([row["current_name"] for row in chunk])
        for row, opsin_smiles in zip(chunk, decoded, strict=True):
            original = row["standardized_smiles"] or None
            roundtrip = standardize_mol(opsin_smiles) if opsin_smiles else None
            matched = original is not None and roundtrip is not None and original == roundtrip
            row.update(
                current_status="matched" if matched else "failed",
                opsin_smiles=opsin_smiles,
                standardized_original=original,
                standardized_roundtrip=roundtrip,
            )


def check_regression(
    baseline_path: Path,
    *,
    processes: int | None,
    chunksize: int,
    opsin_chunk_size: int,
    limit: int = 0,
) -> dict[str, Any]:
    """Run the delta-aware regression check and return a JSON-ready report."""

    from openclatura import name_many
    from openclatura.utils import standardize_mol

    rows, baseline_failure_ids, stored_summary = load_baseline(baseline_path, limit=limit)
    started = time.perf_counter()
    # Keep generation baseline-compatible: the paper names were produced from
    # the stored SMILES. Standardization belongs only to the structural
    # comparison on both sides of the OPSIN round trip.
    generation_smiles = [row["smiles"] for row in rows]
    standardized_smiles = [standardize_mol(smiles) or "" for smiles in generation_smiles]
    results = name_many(
        generation_smiles,
        processes=processes,
        chunksize=chunksize,
    )

    changed: list[dict[str, Any]] = []
    unchanged_matches = 0
    for row, normalized_smiles, result in zip(rows, standardized_smiles, results, strict=True):
        row_id = _row_id(row)
        baseline_failed = row_id in baseline_failure_ids
        current_name = result.name or ""
        baseline_name = row.get(NAME_KEY) or ""
        if current_name == baseline_name:
            unchanged_matches += int(not baseline_failed)
            continue
        changed.append(
            {
                "index": row["index"],
                "smiles": row["smiles"],
                "standardized_smiles": normalized_smiles,
                "baseline_name": baseline_name,
                "baseline_status": "failed" if baseline_failed else "matched",
                "current_name": current_name,
                "current_status": "failed" if not current_name else "pending_opsin",
                "naming_error": result.error,
            }
        )

    if any(row["current_status"] == "pending_opsin" for row in changed):
        verify_changed_rows(changed, opsin_chunk_size)

    current_changed_matches = sum(row["current_status"] == "matched" for row in changed)
    current_matches = unchanged_matches + current_changed_matches
    selected_failure_ids = {_row_id(row) for row in rows} & baseline_failure_ids
    baseline_matches = len(rows) - len(selected_failure_ids)
    passed = current_matches >= baseline_matches

    return {
        "baseline_file": str(baseline_path),
        "generation_input": "stored_smiles",
        "comparison_method": "standardize_mol(original) == standardize_mol(OPSIN(name))",
        "rows": len(rows),
        "baseline_matches": baseline_matches,
        "current_matches": current_matches,
        "match_delta": current_matches - baseline_matches,
        "baseline_accuracy": round(100.0 * baseline_matches / len(rows), 3) if rows else 0.0,
        "current_accuracy": round(100.0 * current_matches / len(rows), 3) if rows else 0.0,
        "changed_names": len(changed),
        "changed_baseline_matches": sum(row["baseline_status"] == "matched" for row in changed),
        "changed_current_matches": current_changed_matches,
        "passed": passed,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "stored_summary": stored_summary,
        "discrepancies": changed,
    }


def _parse_processes(value: str) -> int | None:
    if value == "auto":
        return None
    processes = int(value)
    if processes < 1:
        raise argparse.ArgumentTypeError("processes must be a positive integer or 'auto'")
    return processes


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("baseline", type=Path, help="precomputed *_openclatura.jsonl file")
    parser.add_argument("--processes", type=_parse_processes, default=1, help="worker count or 'auto' (default: 1)")
    parser.add_argument("--chunksize", type=int, default=64, help="naming multiprocessing chunk size")
    parser.add_argument("--opsin-chunk-size", type=int, default=1000, help="changed names per OPSIN invocation")
    parser.add_argument("--limit", type=int, default=0, help="check only the first N rows (development only)")
    parser.add_argument("--report", type=Path, help="write the full JSON report to this path")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = check_regression(
        args.baseline,
        processes=args.processes,
        chunksize=args.chunksize,
        opsin_chunk_size=args.opsin_chunk_size,
        limit=args.limit,
    )
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    status = "PASS" if report["passed"] else "FAIL"
    print(
        f"[{status}] {args.baseline.name}: {report['current_matches']}/{report['rows']} "
        f"valid (baseline {report['baseline_matches']}/{report['rows']}, "
        f"delta {report['match_delta']:+d}); {report['changed_names']} names changed; "
        f"{report['elapsed_seconds']:.1f}s"
    )
    if report["changed_names"]:
        print(f"Changed-name details: {args.report or 'JSON report not requested'}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
