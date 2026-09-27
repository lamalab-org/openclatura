"""Delta-aware paper evaluation checks only changed generated names."""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import pytest


def _load_checker():
    path = Path(__file__).parents[2] / "evaluations" / "check_regression.py"
    spec = importlib.util.spec_from_file_location("check_regression", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CHECKER = _load_checker()


def _baseline(tmp_path: Path, names: list[str], failed: set[int] | None = None) -> Path:
    failed = failed or set()
    baseline = tmp_path / "sample_openclatura.jsonl"
    rows = [{"index": index, "smiles": "CCO", "openclatura_iupac": name} for index, name in enumerate(names)]
    baseline.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    summary, failures = CHECKER._companion_paths(baseline)
    summary.write_text(
        json.dumps({"rows": len(rows), "matches": len(rows) - len(failed), "failures": len(failed)}),
        encoding="utf-8",
    )
    with failures.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["index"])
        writer.writeheader()
        writer.writerows({"index": index} for index in sorted(failed))
    return baseline


def test_unchanged_cached_names_do_not_invoke_opsin(tmp_path, monkeypatch):
    baseline = _baseline(tmp_path, ["ethanol", "ethanol"])
    monkeypatch.setattr(
        "openclatura.name_many",
        lambda *_args, **_kwargs: [
            type("Result", (), {"name": "ethanol", "error": None})(),
            type("Result", (), {"name": "ethanol", "error": None})(),
        ],
    )
    monkeypatch.setattr(CHECKER, "verify_changed_rows", lambda *_args: pytest.fail("OPSIN was called"))

    report = CHECKER.check_regression(baseline, processes=1, chunksize=1, opsin_chunk_size=10)

    assert report["changed_names"] == 0
    assert report["current_matches"] == report["baseline_matches"] == 2


def test_only_changed_names_are_sent_to_opsin(tmp_path, monkeypatch):
    baseline = _baseline(tmp_path, ["ethanol", "old-name", "ethanol"], failed={2})
    current = ["ethanol", "new-name", "ethanol"]
    monkeypatch.setattr(
        "openclatura.name_many",
        lambda *_args, **_kwargs: [type("Result", (), {"name": name, "error": None})() for name in current],
    )
    seen = []

    def verify(rows, _chunk_size):
        seen.extend(row["index"] for row in rows)
        for row in rows:
            row["current_status"] = "matched"

    monkeypatch.setattr(CHECKER, "verify_changed_rows", verify)

    report = CHECKER.check_regression(baseline, processes=1, chunksize=1, opsin_chunk_size=10)

    assert seen == [1]
    assert report["changed_names"] == 1
    assert report["current_matches"] == 2
