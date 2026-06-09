"""Strict golden-name comparison across RDKit versions.

The naming engine reads aromaticity / H-counts / kekulisation off RDKit,
so the same SMILES can in principle name to a different string under
different RDKit versions. This test pins the expected name for every
entry in ``tests/fixtures/diverse_corpus.csv`` to its golden value in
``tests/fixtures/diverse_corpus.golden.json``. Any divergence is
reported as an actionable list.

Run::

    pytest -m golden            # this test only
    pytest                      # default suite *excludes* it

The companion ``rdkit-compat`` CI job runs this test against a matrix
of RDKit versions; failures pinpoint which structures depend on RDKit.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import rdkit

from bluenamer import name_smiles

GOLDEN = Path(__file__).resolve().parent.parent / "fixtures" / "diverse_corpus.golden.json"

pytestmark = pytest.mark.golden


@pytest.fixture(scope="module")
def goldens() -> list[dict]:
    return json.loads(GOLDEN.read_text())


def test_golden_file_is_non_trivial(goldens):
    assert len(goldens) >= 50, "Corpus shrank below 50 entries; regenerate goldens?"


def test_corpus_names_match_golden(goldens, capsys):
    """Every (SMILES → expected name) must match the current namer output."""

    mismatches: list[tuple[str, str, str, str]] = []
    for row in goldens:
        actual = name_smiles(row["smiles"])
        if actual != row["name"]:
            mismatches.append((row["smiles"], row["category"], row["name"], actual))

    if mismatches:
        with capsys.disabled():
            print(f"\nrdkit version: {rdkit.__version__}")
            print(f"divergent SMILES: {len(mismatches)}/{len(goldens)}\n")
            for smi, cat, want, got in mismatches:
                print(f"  [{cat:18s}] {smi}")
                print(f"    expected: {want}")
                print(f"    got:      {got}")
        pytest.fail(
            f"{len(mismatches)} corpus entries name differently on rdkit "
            f"{rdkit.__version__} than the committed goldens"
        )
