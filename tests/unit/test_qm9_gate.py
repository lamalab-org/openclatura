"""Offline contract tests for the dataset gate; never run naming or OPSIN."""

import importlib.util
from pathlib import Path

import pytest

from openclatura.engine import NamingResult
from openclatura.opsin_verify import OpsinCheck


@pytest.fixture
def gate():
    path = Path(__file__).parents[1] / "datasets" / "test_qm9_sample.py"
    spec = importlib.util.spec_from_file_location("qm9_gate_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_public_fixture_preserves_smiles_sequence(gate):
    records = [(17, "CCO"), (83, "CCO")]
    assert gate.qm9_sample.__wrapped__(records) == ["CCO", "CCO"]


def test_empty_sample_fails(gate):
    with pytest.raises(AssertionError, match="nonempty"):
        gate._assert_named([], [])


@pytest.mark.parametrize("size", [0, 2])
def test_result_cardinality_is_strict(gate, size):
    with pytest.raises(AssertionError, match="result count") as exc:
        gate._assert_named([(17, "CCO")], [NamingResult("ethanol")] * size)
    assert "17" in str(exc.value)
    assert "CCO" in str(exc.value)


@pytest.mark.parametrize("failed", [None, NamingResult(""), NamingResult("ethanol", error="failed")])
def test_naming_failure_identifies_duplicate_input_by_index(gate, failed):
    with pytest.raises(AssertionError, match="naming failure") as exc:
        gate._assert_named([(17, "CCO"), (83, "CCO")], [NamingResult("ethanol"), failed])
    assert "index=83 smiles='CCO'" in str(exc.value)
    assert "index=17" not in str(exc.value)


def test_roundtrip_checks_naming_before_opsin(gate, monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("OPSIN must not be called after a naming failure")

    monkeypatch.setattr(gate, "verify_with_opsin", unexpected)
    with pytest.raises(AssertionError, match="naming failure"):
        gate._assert_exact_roundtrips([(17, "CCO")], [NamingResult("")])


def test_exact_match_uses_original_input_and_disables_standardization(gate, monkeypatch):
    def verify(name, smiles, *, standardize_smiles):
        assert (name, smiles, standardize_smiles) == ("ethanol", "CCO", False)
        return OpsinCheck("matched", name, opsin_smiles="OCC")

    monkeypatch.setattr(gate, "verify_with_opsin", verify)
    counts = gate._assert_exact_roundtrips([(17, "CCO")], [NamingResult("ethanol", smiles="C")])
    assert counts == {"exact_matched": 1}


@pytest.mark.parametrize("status", ["mismatched", "name_unparseable", "name_empty", "error", "unexpected"])
def test_nonmatch_statuses_fail_closed(gate, monkeypatch, status):
    monkeypatch.setattr(
        gate, "verify_with_opsin", lambda *a, **kw: OpsinCheck(status, "ethanol", error_message="detail")
    )
    with pytest.raises(AssertionError, match=status) as exc:
        gate._assert_exact_roundtrips([(17, "CCO")], [NamingResult("ethanol")])
    assert "index=17 smiles='CCO'" in str(exc.value)
    assert "detail" in str(exc.value)


@pytest.mark.parametrize(
    ("original", "decoded", "expected"),
    [
        ("CCO", "C", "exact_mismatched"),
        ("CC=O", "C=CO", "exact_mismatched"),  # Tautomer equivalence is not exact.
        ("F[C@H](Cl)Br", "F[C@@H](Cl)Br", "exact_mismatched"),
        ("CCO", None, "exact_parseerror"),
        ("not a SMILES", "CCO", "exact_parseerror"),
    ],
)
def test_matched_status_cannot_bypass_exact_comparison(gate, monkeypatch, original, decoded, expected):
    monkeypatch.setattr(gate, "verify_with_opsin", lambda *a, **kw: OpsinCheck("matched", "name", opsin_smiles=decoded))
    with pytest.raises(AssertionError, match=expected) as exc:
        gate._assert_exact_roundtrips([(17, original)], [NamingResult("name")])
    assert f"index=17 smiles={original!r}" in str(exc.value)


@pytest.mark.parametrize("status", ["skipped_no_opsin", "skipped_no_java"])
def test_missing_dependency_skips_instead_of_passing(gate, monkeypatch, status):
    monkeypatch.setattr(gate, "verify_with_opsin", lambda *a, **kw: OpsinCheck(status, "ethanol"))
    with pytest.raises(pytest.skip.Exception, match="index=17 smiles='CCO'"):
        gate._assert_exact_roundtrips([(17, "CCO")], [NamingResult("ethanol")])


@pytest.mark.parametrize("statuses", [("skipped_no_java", "mismatched"), ("mismatched", "skipped_no_java")])
def test_dependency_skip_cannot_mask_a_failure(gate, monkeypatch, statuses):
    checks = iter(OpsinCheck(status, "ethanol") for status in statuses)
    monkeypatch.setattr(gate, "verify_with_opsin", lambda *a, **kw: next(checks))
    with pytest.raises(AssertionError, match="mismatched"):
        gate._assert_exact_roundtrips([(17, "CCO"), (83, "CCO")], [NamingResult("ethanol")] * 2)


def test_verifier_exception_has_input_context(gate, monkeypatch):
    def verify(*args, **kwargs):
        raise RuntimeError("broken verifier")

    monkeypatch.setattr(gate, "verify_with_opsin", verify)
    with pytest.raises(AssertionError, match="index=17 smiles='CCO'.*RuntimeError: broken verifier"):
        gate._assert_exact_roundtrips([(17, "CCO")], [NamingResult("ethanol")])
