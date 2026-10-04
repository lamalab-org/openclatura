#!/usr/bin/env python3
"""Compare openclatura's internal reconstruction audit with cached OPSIN results.

The input files are the committed ``*_openclatura.jsonl`` evaluation baselines.
Their companion OPSIN failure CSVs provide the independent comparison outcome.
Each molecule is named again with ``verify_self=True`` so the internal audit can
capture and rebuild the name-level assembly plan.

Outputs are resumable per shard:

* ``*_internal_verification.json``: counts, confusion matrix, and examples.
* ``*_internal_cases.csv.gz``: every row where the two verifiers do not both
  positively confirm the generated name.
* ``internal_verification_summary.json``: aggregate report across all shards.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import re
import time
from collections import Counter, defaultdict
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

NAME_KEY = "openclatura_iupac"
VERDICTS = ("confirmed", "mismatch", "abstained", "error")
CASE_FIELDS = (
    "dataset",
    "shard",
    "index",
    "smiles",
    "cached_name",
    "current_name",
    "name_changed",
    "opsin_status",
    "internal_verdict",
    "category",
    "internal_reason",
    "reference_smiles",
    "reconstructed_smiles",
    "naming_error",
)
NAME_CHANGED_CATEGORY = "name_changed_during_internal_audit"


def _read_failure_ids(path: Path) -> set[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {str(row["index"]) for row in csv.DictReader(handle)}


def _row_chunks(path: Path, failure_ids: set[str], chunk_size: int, limit: int) -> Iterable[list[dict[str, Any]]]:
    chunk: list[dict[str, Any]] = []
    seen = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            chunk.append(
                {
                    "index": row["index"],
                    "smiles": row["smiles"],
                    "cached_name": row.get(NAME_KEY) or "",
                    "opsin_status": "failed" if str(row["index"]) in failure_ids else "matched",
                }
            )
            seen += 1
            if len(chunk) == chunk_size:
                yield chunk
                chunk = []
            if limit and seen >= limit:
                break
    if chunk:
        yield chunk


def _audit_chunk(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    from openclatura import name

    output = []
    for row in rows:
        result = name(row["smiles"], verify_self=True)
        audit = result.self_audit
        if audit is None:
            verdict = "error"
            reason = result.error or "naming produced no internal audit"
            reference = ""
            reconstructed = ""
        else:
            verdict = audit.verdict
            reason = audit.reason
            reference = audit.reference_smiles or ""
            reconstructed = audit.reconstructed_smiles or ""
        output.append(
            {
                **row,
                "current_name": result.name or "",
                "name_changed": result.name != row["cached_name"],
                "internal_verdict": verdict,
                "internal_reason": reason,
                "reference_smiles": reference,
                "reconstructed_smiles": reconstructed,
                "naming_error": result.error or "",
            }
        )
    return output


def category(opsin_status: str, internal_verdict: str) -> str:
    """Return the comparison category for one OPSIN/internal outcome pair."""

    if opsin_status == "matched":
        return {
            "confirmed": "agree_valid",
            "mismatch": "internal_rejected_opsin_match",
            "abstained": "internal_abstained_on_opsin_match",
            "error": "internal_error_on_opsin_match",
        }[internal_verdict]
    return {
        "confirmed": "missed_opsin_failure",
        "mismatch": "caught_opsin_failure",
        "abstained": "internal_abstained_on_opsin_failure",
        "error": "internal_error_on_opsin_failure",
    }[internal_verdict]


def _reason_group(reason: str) -> str:
    if not reason:
        return ""
    if "reconstructed " in reason and " != input " in reason:
        return "reconstructed graph differs from input"
    if "unnamed atoms:" in reason:
        return "unnamed atoms"
    if re.search(r"(?:substituent|ester group|principal-suffix modifier) .+ not modelled", reason):
        family = re.search(r"(substituent|ester group|principal-suffix modifier)", reason).group(1)
        return f"{family} not modelled"
    if re.search(r"indicated hydrogen [^ ]+ not placeable", reason):
        return "indicated hydrogen not placeable"
    if re.search(r"(?:substituent|principal-group|parent charge) locant .+ outside parent", reason):
        family = re.search(r"(substituent|principal-group|parent charge) locant", reason).group(1)
        return f"{family} locant outside parent"
    reason = re.sub(r"\b\d+:[A-Z][a-z]?\b", "<atom>", reason)
    return reason[:240]


def _metrics(matrix: dict[str, dict[str, int]]) -> dict[str, float | int | None]:
    matched = matrix["matched"]
    failed = matrix["failed"]
    caught = failed["mismatch"]
    missed = failed["confirmed"]
    false_alarm = matched["mismatch"]
    decisive = sum(row["confirmed"] + row["mismatch"] for row in matrix.values())
    total = sum(sum(row.values()) for row in matrix.values())
    mismatch_total = caught + false_alarm
    return {
        "total": total,
        "decisive": decisive,
        "decisive_coverage_percent": round(100.0 * decisive / total, 6) if total else None,
        "opsin_failures_caught": caught,
        "opsin_failures_missed": missed,
        "failure_detection_recall_percent": round(100.0 * caught / sum(failed.values()), 6)
        if sum(failed.values())
        else None,
        "mismatch_precision_percent": round(100.0 * caught / mismatch_total, 6) if mismatch_total else None,
        "opsin_matches_rejected": false_alarm,
    }


def _empty_matrix() -> dict[str, dict[str, int]]:
    return {status: {verdict: 0 for verdict in VERDICTS} for status in ("matched", "failed")}


def _score_file(
    path: Path,
    output_dir: Path,
    pool: ProcessPoolExecutor,
    *,
    chunk_size: int,
    limit: int,
    examples_per_category: int,
) -> dict[str, Any]:
    failures_path = path.with_name(f"{path.stem}_opsin_failures.csv")
    if not failures_path.is_file():
        raise FileNotFoundError(f"missing OPSIN failure cache: {failures_path}")

    dataset = path.parent.name
    shard = path.stem.removesuffix("_openclatura")
    report_path = output_dir / f"{shard}_internal_verification.json"
    cases_path = output_dir / f"{shard}_internal_cases.csv.gz"
    cases_tmp = cases_path.with_suffix(cases_path.suffix + ".tmp")
    failure_ids = _read_failure_ids(failures_path)
    matrix = _empty_matrix()
    categories: Counter[str] = Counter()
    reasons: dict[str, Counter[str]] = defaultdict(Counter)
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rows = changed_names = 0
    started = time.perf_counter()

    payloads = _row_chunks(path, failure_ids, chunk_size, limit)
    with gzip.open(cases_tmp, "wt", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CASE_FIELDS, lineterminator="\n")
        writer.writeheader()
        for audited_chunk in pool.map(_audit_chunk, payloads):
            for record in audited_chunk:
                rows += 1
                verdict = record["internal_verdict"]
                status = record["opsin_status"]
                changed = bool(record["name_changed"])
                comparison = NAME_CHANGED_CATEGORY if changed else category(status, verdict)
                record.update(dataset=dataset, shard=shard, category=comparison)
                categories[comparison] += 1
                changed_names += int(changed)
                reasons[verdict][_reason_group(record["internal_reason"])] += 1
                if changed:
                    writer.writerow({field: record.get(field, "") for field in CASE_FIELDS})
                    if len(examples[comparison]) < examples_per_category:
                        examples[comparison].append({field: record.get(field, "") for field in CASE_FIELDS})
                    continue
                matrix[status][verdict] += 1
                if comparison != "agree_valid" or record["name_changed"]:
                    writer.writerow({field: record.get(field, "") for field in CASE_FIELDS})
                    if len(examples[comparison]) < examples_per_category:
                        examples[comparison].append({field: record.get(field, "") for field in CASE_FIELDS})

    cases_tmp.replace(cases_path)
    report = {
        "dataset": dataset,
        "shard": shard,
        "source": str(path),
        "opsin_failures_source": str(failures_path),
        "rows": rows,
        "comparable_rows": rows - changed_names,
        "name_changes_from_cache": changed_names,
        "matrix": matrix,
        "categories": dict(sorted(categories.items())),
        "metrics": _metrics(matrix),
        "reason_counts": {verdict: dict(counter.most_common()) for verdict, counter in sorted(reasons.items())},
        "examples": dict(sorted(examples.items())),
        "cases_file": str(cases_path),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"{dataset}/{shard}: rows={rows} confirmed={matrix['matched']['confirmed'] + matrix['failed']['confirmed']} "
        f"mismatch={matrix['matched']['mismatch'] + matrix['failed']['mismatch']} "
        f"abstained={matrix['matched']['abstained'] + matrix['failed']['abstained']} "
        f"errors={matrix['matched']['error'] + matrix['failed']['error']} "
        f"({report['elapsed_seconds']:.1f}s)",
        flush=True,
    )
    return report


def _aggregate(reports: list[dict[str, Any]]) -> dict[str, Any]:
    matrix = _empty_matrix()
    categories: Counter[str] = Counter()
    datasets: dict[str, dict[str, Any]] = {}
    reason_counts: dict[str, Counter[str]] = defaultdict(Counter)
    changed_names = 0
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for report in reports:
        changed_names += report["name_changes_from_cache"]
        categories.update(report["categories"])
        for status in matrix:
            for verdict in VERDICTS:
                matrix[status][verdict] += report["matrix"][status][verdict]
        for verdict, counts in report["reason_counts"].items():
            reason_counts[verdict].update(counts)
        for comparison, rows in report["examples"].items():
            examples[comparison].extend(rows)

        dataset = datasets.setdefault(
            report["dataset"],
            {"rows": 0, "comparable_rows": 0, "name_changes_from_cache": 0, "matrix": _empty_matrix()},
        )
        dataset["rows"] += report["rows"]
        dataset["comparable_rows"] += report["comparable_rows"]
        dataset["name_changes_from_cache"] += report["name_changes_from_cache"]
        for status in matrix:
            for verdict in VERDICTS:
                dataset["matrix"][status][verdict] += report["matrix"][status][verdict]

    for dataset in datasets.values():
        dataset["metrics"] = _metrics(dataset["matrix"])

    return {
        "rows": sum(report["rows"] for report in reports),
        "comparable_rows": sum(report["comparable_rows"] for report in reports),
        "shards": len(reports),
        "name_changes_from_cache": changed_names,
        "matrix": matrix,
        "categories": dict(sorted(categories.items())),
        "metrics": _metrics(matrix),
        "datasets": dict(sorted(datasets.items())),
        "reason_counts": {verdict: dict(counter.most_common()) for verdict, counter in sorted(reason_counts.items())},
        "examples": {comparison: rows[:20] for comparison, rows in sorted(examples.items())},
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="*", type=Path, help="cached *_openclatura.jsonl files")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evaluations/results/internal_verification"),
    )
    parser.add_argument("--workers", type=int, default=min(12, os.cpu_count() or 1))
    parser.add_argument("--chunk-size", type=int, default=250)
    parser.add_argument("--limit", type=int, default=0, help="rows per shard; 0 evaluates all")
    parser.add_argument("--examples-per-category", type=int, default=5)
    parser.add_argument("--resume", action="store_true", help="reuse completed per-shard reports")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    inputs = args.inputs or sorted(Path("evaluations/results").glob("*/*_openclatura.jsonl"))
    if not inputs:
        raise SystemExit("no openclatura evaluation baselines found")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reports = []

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for path in inputs:
            shard = path.stem.removesuffix("_openclatura")
            report_path = args.output_dir / f"{shard}_internal_verification.json"
            cases_path = args.output_dir / f"{shard}_internal_cases.csv.gz"
            if args.resume and report_path.is_file() and cases_path.is_file():
                report = json.loads(report_path.read_text(encoding="utf-8"))
                if not args.limit or report["rows"] == args.limit:
                    reports.append(report)
                    print(f"reuse {report['dataset']}/{shard}", flush=True)
                    continue
            reports.append(
                _score_file(
                    path,
                    args.output_dir,
                    pool,
                    chunk_size=args.chunk_size,
                    limit=args.limit,
                    examples_per_category=args.examples_per_category,
                )
            )

    summary = _aggregate(reports)
    summary["inputs"] = [str(path) for path in inputs]
    summary["limit_per_shard"] = args.limit
    summary_path = args.output_dir / "internal_verification_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary["metrics"], indent=2))
    print(f"wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
