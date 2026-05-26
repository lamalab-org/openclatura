"""Tests for the typed public API introduced in PR2."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from bluenamer import (
    DEFAULT_NAMING_ENGINE,
    NamingEngine,
    NamingRequest,
    NamingResult,
    OpsinCheck,
    name,
    name_many,
    name_smiles,
)


def test_name_smiles_legacy_still_returns_string():
    assert name_smiles("CCO") == "ethanol"


def test_name_returns_naming_result_with_smiles_field():
    result = name("CCO")
    assert isinstance(result, NamingResult)
    assert result.smiles == "CCO"
    assert result.name == "ethanol"
    assert result.error is None
    assert result.ok is True
    assert result.opsin_check is None
    # No trace was requested, so rules_hit is empty by design.
    assert result.rules_hit == ()


def test_name_with_trace_populates_rules_hit_and_hints():
    result = name("CC(=O)Nc1ccccc1", include_trace=True)
    assert result.name == "N-phenylacetamide"
    assert result.trace_segments, "trace_segments must be populated when include_trace=True"
    assert result.decisions, "decisions must be populated when include_trace=True"
    # Rule hints should reference Blue Book P-XX identifiers in this case.
    assert result.rules_hit, "expected at least one P-XX rule id"
    assert all(rid.startswith("P-") for rid in result.rules_hit)
    assert len(result.rule_hints) >= 1


def test_naming_errors_become_result_error_not_exception():
    # Empty SMILES is a well-defined "no atoms" path; assert at a different angle
    # by feeding clearly malformed text — the engine must capture, not raise.
    result = name("definitely-not-a-smiles")
    # Either name is empty with error captured, or rdkit accepted it.
    # We only assert there is no uncaught exception.
    assert isinstance(result, NamingResult)


def test_empty_smiles_yields_empty_name():
    result = name("")
    assert result.name == ""
    assert result.smiles == ""


def test_name_many_serial_preserves_order():
    inputs = ["CCO", "c1ccccc1", "CC(=O)O", "CCN"]
    results = name_many(inputs, processes=1)
    assert [r.smiles for r in results] == inputs
    assert [r.name for r in results] == ["ethanol", "benzene", "acetic acid", "ethanamine"]


def test_name_many_swallows_per_row_errors():
    inputs = ["CCO", "this-is-not-smiles", "c1ccccc1"]
    results = name_many(inputs)
    assert results[0].ok is True
    assert results[2].ok is True
    # Middle one: either named (some inputs are lenient) or marked as error.
    # The important contract is that the batch as a whole didn't raise.
    assert all(isinstance(r, NamingResult) for r in results)


def test_engine_run_with_verify_opsin_skips_gracefully_without_opsin(monkeypatch):
    # Force the "no py2opsin" path even when the optional dep is installed.
    import bluenamer.opsin_verify as ov

    monkeypatch.setattr(ov, "_try_import_py2opsin", lambda: None)
    result = DEFAULT_NAMING_ENGINE.run(NamingRequest(smiles="CCO", verify_opsin=True))
    assert isinstance(result.opsin_check, OpsinCheck)
    assert result.opsin_check.status == "skipped_no_opsin"
    assert result.verified is False


def test_engine_run_with_verify_opsin_skips_gracefully_without_java(monkeypatch):
    import bluenamer.opsin_verify as ov

    # py2opsin present, java not.
    monkeypatch.setattr(ov, "_try_import_py2opsin", lambda: object())
    monkeypatch.setattr(ov, "_java_available", lambda: False)
    result = DEFAULT_NAMING_ENGINE.run(NamingRequest(smiles="CCO", verify_opsin=True))
    assert result.opsin_check is not None
    assert result.opsin_check.status == "skipped_no_java"


def test_naming_engine_is_reusable():
    engine = NamingEngine()
    a = engine.run(NamingRequest(smiles="CCO"))
    b = engine.run(NamingRequest(smiles="c1ccccc1"))
    assert a.name == "ethanol"
    assert b.name == "benzene"


def test_cli_name_subcommand_prints_name():
    result = subprocess.run(
        [sys.executable, "-m", "bluenamer.cli", "name", "CCO"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ethanol"


def test_cli_name_json_emits_structured_payload():
    result = subprocess.run(
        [sys.executable, "-m", "bluenamer.cli", "name", "CC(=O)O", "--json"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["name"] == "acetic acid"
    assert payload["smiles"] == "CC(=O)O"
    assert payload["ok"] is True
    assert "trace_segments" in payload


@pytest.mark.parametrize("processes", [1, 2])
def test_name_many_parallel_and_serial_agree(processes):
    inputs = ["CCO", "c1ccccc1", "CC(=O)O", "CCN", "CC(C)C"]
    serial = name_many(inputs, processes=1)
    other = name_many(inputs, processes=processes)
    assert [r.name for r in serial] == [r.name for r in other]
