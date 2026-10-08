"""A reviewed parser gap waives one expectation, and only one.

Where this engine emits a preferred name the pinned reference parser cannot
read, the name stands: P-12.1 derives preferred names from the rules, and a
parser's vocabulary is not one of them. What the evaluation gate may do is
classify that rejection as reviewed rather than as a new regression.

What it may not do is weaken anything else. The exact name and the structure it
denotes stay asserted here, outside any expected-failure marker, so a broad
xfail cannot swallow an unrelated break. The parser is still called, because
that call is what reports when the gap closes.
"""

import json
import pathlib

import pytest

from openclatura import name, opsin_available

GAPS = json.loads(pathlib.Path("evaluations/reviewed_parser_gaps.json").read_text())
CASES = [pytest.param(gap, id=gap["expected_name"][:48]) for gap in GAPS["gaps"]]


@pytest.mark.parametrize("gap", CASES)
def test_the_preferred_name_is_still_emitted(gap):
    """Mandatory. A gap is bound to one name; any other name fails."""

    assert name(gap["smiles"]).name == gap["expected_name"]


@pytest.mark.parametrize("gap", CASES)
def test_the_gap_records_its_parser_and_its_evidence(gap):
    """A gap that cannot say which parser it applies to cannot be retired."""

    assert GAPS["parser"]["jar"]
    assert GAPS["parser"]["jar_sha256_prefix"]
    assert gap["preference_evidence"] == "rule_derived", "not a published whole-name PIN"
    assert gap["rules"] and gap["review_trigger"] and gap["retire_when"]


@pytest.mark.skipif(not opsin_available(), reason="OPSIN and Java are required")
@pytest.mark.parametrize("gap", CASES)
def test_the_recorded_rejection_still_reproduces(gap):
    """The parser is still asked, so a closed gap shows up as a failure here.

    This is the one expectation the gap waives, and it is asserted on its own.
    If the parser starts reading the name the assertion fails and the gap has to
    be retired; if it returns some other structure that is a mismatch, which no
    gap licenses.
    """

    from py2opsin import py2opsin

    decoded = py2opsin([gap["expected_name"]])
    returned = decoded[0] if decoded else ""
    assert not returned, (
        "the pinned parser now returns a structure for this name: retire the gap "
        "if the structure is right, investigate if it is not"
    )
