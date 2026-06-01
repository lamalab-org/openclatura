#!/usr/bin/env python3
"""Validate STOUT JSONL predictions by round-tripping IUPAC names through OPSIN."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
import time
from collections.abc import Iterable, Iterator
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Any

from tqdm import tqdm


_CHEM = None
_PY2OPSIN = None


def _default_workers() -> int:
    return max(1, min(os.cpu_count() or 1, 8))


def _default_failures_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_opsin_failures.csv")


def _first_nonspace_char(path: Path) -> str:
    with path.open(encoding="utf-8") as handle:
        while True:
            char = handle.read(1)
            if not char:
                return ""
            if not char.isspace():
                return char


def _count_rows(path: Path) -> int:
    if _first_nonspace_char(path) == "[":
        with path.open(encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, list):
            raise ValueError(f"{path} must contain a JSON array or JSONL object rows")
        return len(data)

    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _iter_rows(path: Path) -> Iterator[dict[str, Any]]:
    if _first_nonspace_char(path) == "[":
        with path.open(encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, list):
            raise ValueError(f"{path} must contain a JSON array or JSONL object rows")
        for row in data:
            if not isinstance(row, dict):
                raise ValueError("each prediction row must be a JSON object")
            yield row
        return

    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"line {line_number} is not a JSON object")
            yield row


def _iter_batches(rows: Iterable[dict[str, Any]], batch_size: int) -> Iterator[list[dict[str, Any]]]:
    batch: list[dict[str, Any]] = []
    for row in rows:
        batch.append(row)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _load_worker_deps() -> None:
    global _CHEM, _PY2OPSIN
    if _CHEM is not None and _PY2OPSIN is not None:
        return

    import py2opsin
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    _CHEM = Chem
    _PY2OPSIN = py2opsin


def _canonical_smiles(smiles: Any) -> str | None:
    _load_worker_deps()
    if not isinstance(smiles, str) or not smiles:
        return None
    mol = _CHEM.MolFromSmiles(smiles)
    if mol is None:
        return None
    return _CHEM.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def _roundtrip_names(names: list[str]) -> list[str | None]:
    _load_worker_deps()
    tmp_dir = Path(tempfile.gettempdir()) / "py2opsin_parallel"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_dir / f"py2opsin_{os.getpid()}_{time.time_ns()}.txt"
    results = list(_PY2OPSIN.py2opsin(names, tmp_fpath=str(tmp_path)))
    if len(results) != len(names):
        raise RuntimeError(f"OPSIN returned {len(results)} rows for {len(names)} inputs")
    return results


def _validate_batch(payload: tuple[int, list[dict[str, Any]], str, str, str]) -> tuple[int, dict[str, int], list[dict[str, Any]]]:
    chunk_index, rows, smiles_key, name_key, index_key = payload
    names = [str(row.get(name_key) or "") for row in rows]
    opsin_smiles = _roundtrip_names(names)
    counts = {
        "rows": 0,
        "original_valid": 0,
        "opsin_nonempty": 0,
        "opsin_valid": 0,
        "matches": 0,
        "failures": 0,
    }
    failures: list[dict[str, Any]] = []

    for row, reconstructed in zip(rows, opsin_smiles):
        original = row.get(smiles_key)
        name = row.get(name_key)
        original_canon = _canonical_smiles(original)
        reconstructed_canon = _canonical_smiles(reconstructed)

        counts["rows"] += 1
        if original_canon is not None:
            counts["original_valid"] += 1
        if reconstructed:
            counts["opsin_nonempty"] += 1
        if reconstructed_canon is not None:
            counts["opsin_valid"] += 1

        matched = original_canon is not None and original_canon == reconstructed_canon
        if matched:
            counts["matches"] += 1
            continue

        counts["failures"] += 1
        if original_canon is None:
            reason = "original_invalid"
        elif reconstructed_canon is None:
            reason = "opsin_empty_or_invalid"
        else:
            reason = "canonical_mismatch"

        failures.append(
            {
                "index": row.get(index_key),
                "smiles": original,
                "original_canon": original_canon,
                "stout_iupac": name,
                "opsin_smiles": reconstructed,
                "opsin_canon": reconstructed_canon,
                "reason": reason,
            }
        )

    return chunk_index, counts, failures


def _payloads(
    rows: Iterable[dict[str, Any]],
    *,
    chunk_size: int,
    smiles_key: str,
    name_key: str,
    index_key: str,
) -> Iterator[tuple[int, list[dict[str, Any]], str, str, str]]:
    for chunk_index, batch in enumerate(_iter_batches(rows, chunk_size)):
        yield chunk_index, batch, smiles_key, name_key, index_key


def _add_counts(total: dict[str, int], batch: dict[str, int]) -> None:
    for key, value in batch.items():
        total[key] += value


def _write_result(
    result: tuple[int, dict[str, int], list[dict[str, Any]]],
    *,
    writer: csv.DictWriter,
    progress: tqdm,
    counts: dict[str, int],
) -> None:
    _, batch_counts, failures = result
    _add_counts(counts, batch_counts)
    writer.writerows(failures)
    progress.update(batch_counts["rows"])


def _drain_one(
    futures: set[Future],
    *,
    writer: csv.DictWriter,
    progress: tqdm,
    counts: dict[str, int],
) -> None:
    done, _ = wait(futures, return_when=FIRST_COMPLETED)
    for future in done:
        futures.remove(future)
        _write_result(future.result(), writer=writer, progress=progress, counts=counts)


def _run_validation(
    payload_iter: Iterable[tuple[int, list[dict[str, Any]], str, str, str]],
    *,
    workers: int,
    writer: csv.DictWriter,
    progress: tqdm,
    counts: dict[str, int],
) -> None:
    if workers == 1:
        for payload in payload_iter:
            _write_result(_validate_batch(payload), writer=writer, progress=progress, counts=counts)
        return

    max_in_flight = workers * 2
    with ProcessPoolExecutor(max_workers=workers, initializer=_load_worker_deps) as executor:
        futures: set[Future] = set()
        for payload in payload_iter:
            futures.add(executor.submit(_validate_batch, payload))
            if len(futures) >= max_in_flight:
                _drain_one(futures, writer=writer, progress=progress, counts=counts)
        while futures:
            _drain_one(futures, writer=writer, progress=progress, counts=counts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate STOUT prediction JSONL by converting stout_iupac through OPSIN and comparing canonical SMILES."
    )
    parser.add_argument("input", type=Path, help="JSONL file with smiles and stout_iupac fields")
    parser.add_argument("--failures-csv", type=Path, help="where to write mismatches")
    parser.add_argument("--summary-json", type=Path, help="optional path for machine-readable summary")
    parser.add_argument("--chunk-size", type=int, default=2000, help="number of names to send to OPSIN per worker call")
    parser.add_argument("--workers", type=int, default=_default_workers(), help="parallel OPSIN/RDKit worker processes")
    parser.add_argument("--limit", type=int, default=0, help="number of rows to validate; 0 means all")
    parser.add_argument("--smiles-key", default="smiles")
    parser.add_argument("--name-key", default="stout_iupac")
    parser.add_argument("--index-key", default="index")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args(argv)

    if args.chunk_size < 1:
        parser.error("--chunk-size must be >= 1")
    if args.workers < 1:
        parser.error("--workers must be >= 1")
    if args.limit < 0:
        parser.error("--limit must be >= 0")
    if not args.input.exists():
        parser.error(f"input file does not exist: {args.input}")

    failures_csv = args.failures_csv or _default_failures_path(args.input)
    total_rows = _count_rows(args.input)
    if args.limit:
        total_rows = min(total_rows, args.limit)

    counts = {
        "rows": 0,
        "original_valid": 0,
        "opsin_nonempty": 0,
        "opsin_valid": 0,
        "matches": 0,
        "failures": 0,
    }
    fieldnames = [
        "index",
        "smiles",
        "original_canon",
        "stout_iupac",
        "opsin_smiles",
        "opsin_canon",
        "reason",
    ]

    start = time.perf_counter()
    row_iter = _iter_rows(args.input)
    if args.limit:
        row_iter = (row for _, row in zip(range(args.limit), row_iter))
    payload_iter = _payloads(
        row_iter,
        chunk_size=args.chunk_size,
        smiles_key=args.smiles_key,
        name_key=args.name_key,
        index_key=args.index_key,
    )

    with failures_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        progress = tqdm(
            total=total_rows,
            desc="OPSIN",
            unit="mol",
            disable=args.no_progress,
            file=sys.stderr,
        )
        try:
            _run_validation(
                payload_iter,
                workers=args.workers,
                writer=writer,
                progress=progress,
                counts=counts,
            )
        finally:
            progress.close()

    elapsed = time.perf_counter() - start
    accuracy = counts["matches"] / counts["rows"] * 100 if counts["rows"] else 0.0
    rate = counts["rows"] / elapsed if elapsed else 0.0
    summary = {
        **counts,
        "accuracy": accuracy,
        "elapsed_seconds": elapsed,
        "rows_per_second": rate,
        "workers": args.workers,
        "chunk_size": args.chunk_size,
        "failures_csv": str(failures_csv),
    }

    if args.summary_json:
        args.summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"rows={counts['rows']}")
    print(f"original_valid={counts['original_valid']}")
    print(f"opsin_nonempty={counts['opsin_nonempty']}")
    print(f"opsin_valid={counts['opsin_valid']}")
    print(f"matches={counts['matches']}")
    print(f"failures={counts['failures']}")
    print(f"accuracy={accuracy:.2f}%")
    print(f"elapsed={elapsed:.2f}s")
    print(f"rate={rate:.2f}/s")
    print(f"workers={args.workers}")
    print(f"chunk_size={args.chunk_size}")
    print(f"failure_csv={failures_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
