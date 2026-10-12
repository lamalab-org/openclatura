"""A reviewed numbering mismatch is a different exception from a reviewed gap.

A parser that rejects a name has told us only that its vocabulary is narrower
than the rules. A parser that accepts the name and builds a different molecule
has made a claim about the numbering of the ring system, which is the very thing
in dispute. The second case therefore carries a higher burden: the numbered
parent graph has to come from somewhere other than the engine whose map produced
the name, because a reconstruction that consumes that same map shows only
internal consistency.

What a record may do is let the release gate call the outcome reviewed. What it
may not do is make it a round trip. The raw outcome stays mismatched, and the
record is honoured only while the pinned parser keeps returning exactly the
interpretation that was reviewed - a correct reading, a different wrong reading
or a rejection all fall outside it and surface as failures.
"""

import json
import pathlib
import sys

import pytest

from openclatura import name, opsin_available

sys.path.insert(0, str(pathlib.Path("evaluations").resolve()))
import check_regression  # noqa: E402

PATH = pathlib.Path("evaluations/reviewed_parser_numbering_mismatches.json")
DATA = json.loads(PATH.read_text())
GAPS = json.loads(pathlib.Path("evaluations/reviewed_parser_gaps.json").read_text())
CASES = [pytest.param(record, id=record["expected_name"][:48]) for record in DATA["mismatches"]]


def test_the_two_categories_pin_the_same_parser():
    """Two files, one reviewed environment: a record is bound to that artifact."""

    assert DATA["parser"] == GAPS["parser"]
    assert DATA["parser"]["jar"] and DATA["parser"]["jar_sha256_prefix"]


def test_the_categories_stay_distinct():
    """A rejection waiver must never be reused to excuse a mismatch."""

    rejections = {(gap["smiles"], gap["expected_name"]) for gap in GAPS["gaps"]}
    numbering = {(row["smiles"], row["expected_name"]) for row in DATA["mismatches"]}
    assert not rejections & numbering


def test_the_schema_demands_independent_numbering_evidence():
    """The independently derived parent graph is what makes a record admissible."""

    required = set(DATA["required_fields"])
    assert {"disputed_numbering", "drawing_record", "independent_source"} <= required
    assert "derivative_reconstruction" in required
    assert "parser_returned_standardized" in required


def test_an_incomplete_record_is_refused(tmp_path):
    """A record missing its evidence is a loading error, not a silent waiver."""

    path = tmp_path / "mismatches.json"
    path.write_text(
        json.dumps(
            {
                "required_fields": DATA["required_fields"],
                "mismatches": [{"smiles": "c1ccccc1", "expected_name": "benzene"}],
            }
        )
    )
    with pytest.raises(RuntimeError, match="missing"):
        check_regression._reviewed_numbering_mismatches(path)


def _row(smiles, nm, standardized):
    return {
        "smiles": smiles,
        "current_name": nm,
        "standardized_smiles": standardized,
        "current_status": "pending_opsin",
    }


@pytest.mark.parametrize(
    "returned,expected_status,expected_flag",
    [
        ("CCO", "reviewed_numbering_mismatch", None),
        ("OCC", "reviewed_numbering_mismatch", None),
        ("CCN", "failed", "different_interpretation"),
        ("", "failed", "unexpected_rejection"),
        # A parser that starts reading the name correctly really is a match;
        # the record is stale, and the gate fails on it until it is retired.
        ("CCC", "matched", "stale"),
    ],
)
def test_only_the_reviewed_interpretation_is_honoured(monkeypatch, tmp_path, returned, expected_status, expected_flag):
    """Anything but the recorded wrong reading falls outside the record."""

    from openclatura.utils import standardize_mol

    target, wrong = "CCC", "CCO"
    path = tmp_path / "mismatches.json"
    path.write_text(
        json.dumps(
            {
                "required_fields": DATA["required_fields"],
                "mismatches": [
                    {field: "x" for field in DATA["required_fields"]}
                    | {
                        "smiles": target,
                        "expected_name": "propane",
                        "parser_returned_standardized": standardize_mol(wrong),
                        "preference_evidence": "rule_derived",
                    }
                ],
            }
        )
    )
    loader = check_regression._reviewed_numbering_mismatches
    monkeypatch.setattr(check_regression, "_reviewed_numbering_mismatches", lambda _path=None: loader(path))
    monkeypatch.setattr(check_regression, "_opsin_roundtrip", lambda names: [returned])
    rows = [_row(target, "propane", standardize_mol(target))]
    check_regression.verify_changed_rows(rows, 64)
    assert rows[0]["current_status"] == expected_status
    assert rows[0].get("numbering_mismatch") == expected_flag


@pytest.mark.parametrize("record", CASES)
def test_the_preferred_name_is_still_emitted(record):
    """Mandatory. A record is bound to one name; any other name fails."""

    assert name(record["smiles"]).name == record["expected_name"]


@pytest.mark.skipif(not opsin_available(), reason="OPSIN and Java are required")
@pytest.mark.parametrize("record", CASES)
def test_the_recorded_misreading_still_reproduces(record):
    """The parser is still asked, so a changed answer shows up here as a failure."""

    from py2opsin import py2opsin

    from openclatura.utils import standardize_mol

    decoded = py2opsin([record["expected_name"]])
    returned = decoded[0] if decoded else ""
    assert returned, "the parser now rejects this name: that is outside the record"
    assert standardize_mol(returned) == record["parser_returned_standardized"], (
        "the parser returns a different structure than the one reviewed: re-review rather than carry the record"
    )
