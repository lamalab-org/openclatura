"""Smoke + metric test over a hand-curated diverse corpus.

The corpus lives at ``tests/fixtures/diverse_corpus.csv`` and covers
alkanes, alkenes, alkynes, aromatics, heterocycles, alcohols, ketones,
acids, esters, amines/amides, sugars, fatty acids and a handful of
multi-functional cases. It is intentionally small (~100 entries) so
this test stays in the default pytest run.

The test asserts two things:

1. Every entry must produce a non-empty name without raising.
2. The fraction of entries that produce a non-empty name must stay
   above a floor (currently 0.95). This is a regression guard — drops
   below the floor will fail CI and point at the offending SMILES.

A separate OPSIN round-trip metric is reported when py2opsin + Java are
available, but only logged (not asserted) because round-trip rates are
expected to drift as the namer evolves. The number is captured by
pytest's ``--durations`` / printed summary so trends are still visible
in CI output.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from bluenamer import name as name_one
from bluenamer.opsin_verify import verify_with_opsin

FIXTURE = Path(__file__).resolve().parent.parent / "fixtures" / "diverse_corpus.csv"

# Floor for the basic naming pass rate. Tightened as the namer matures;
# never relaxed silently.
NAME_PASS_RATE_FLOOR = 0.95


def _load_corpus() -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    with FIXTURE.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            smi = row["smiles"].strip()
            cat = row.get("category", "").strip()
            if smi:
                rows.append((smi, cat))
    return rows


@pytest.fixture(scope="session")
def corpus() -> list[tuple[str, str]]:
    rows = _load_corpus()
    assert rows, "Corpus must not be empty"
    return rows


def test_corpus_naming_does_not_raise(corpus):
    failures = []
    for smi, cat in corpus:
        result = name_one(smi)
        if result.error is not None:
            failures.append((smi, cat, result.error))
    assert not failures, f"name_one raised internally on {len(failures)} corpus entries: {failures[:5]}"


def test_corpus_naming_pass_rate(corpus):
    """At least ``NAME_PASS_RATE_FLOOR`` of the corpus must produce a non-empty name."""

    named = []
    unnamed = []
    for smi, cat in corpus:
        result = name_one(smi)
        if result.name:
            named.append((smi, cat, result.name))
        else:
            unnamed.append((smi, cat))

    rate = len(named) / len(corpus)
    if rate < NAME_PASS_RATE_FLOOR:
        # Surface the offenders so the failure is actionable.
        sample = "\n".join(f"  - {s} [{c}]" for s, c in unnamed[:10])
        pytest.fail(
            f"Naming pass rate {rate:.2%} below floor {NAME_PASS_RATE_FLOOR:.2%}.\n" f"First unnamed entries:\n{sample}"
        )


@pytest.mark.opsin
def test_corpus_opsin_roundtrip_rate(corpus, capsys):
    """Report the OPSIN round-trip match rate. Logged only — not asserted.

    Skipped when py2opsin or Java are missing (the helper returns
    ``skipped_*`` statuses we treat as "out of scope" for the metric).
    """

    matched = 0
    mismatched = 0
    skipped = 0
    unparseable = 0
    errored = 0
    for smi, _ in corpus:
        result = name_one(smi)
        if not result.name:
            errored += 1
            continue
        check = verify_with_opsin(result.name, smi)
        if check.status.startswith("skipped_"):
            skipped += 1
        elif check.status == "matched":
            matched += 1
        elif check.status == "mismatched":
            mismatched += 1
        elif check.status == "name_unparseable":
            unparseable += 1
        else:
            errored += 1

    total = len(corpus)
    if skipped == total:
        pytest.skip("OPSIN / Java not available")
    in_scope = total - skipped
    rate = matched / in_scope if in_scope else 0.0
    # Print outside the captured stream so --durations / CI logs see it.
    with capsys.disabled():
        print(
            f"\n[corpus] opsin round-trip: matched={matched} "
            f"mismatched={mismatched} unparseable={unparseable} errored={errored} "
            f"skipped={skipped}/{total} → match rate (in-scope) = {rate:.2%}"
        )
    # Sanity floor: at least *some* round-trips must work, otherwise the
    # whole pipeline is broken and the test should be loud.
    assert matched > 0, "Zero OPSIN matches across the entire corpus — pipeline likely broken"
