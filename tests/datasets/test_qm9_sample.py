"""Opt-in QM9 sample test.

Pulls a small random sample from ``yairschiff/qm9`` and requires every input
to be named and round-trip exactly through OPSIN. Exact means equal RDKit
canonical isomeric SMILES, not standardized/tautomer/resonance equivalence.
Marked ``dataset``/``slow``; run with ``pytest -m dataset``.

Tunable via env: ``OPENCLATURA_DATASET_SAMPLE_N``, ``OPENCLATURA_DATASET_SEED``.
Indexed records are ``(train_row_index, original_smiles)``. There is no
baseline, inferred allowance, or accepted failure budget.
"""

from __future__ import annotations

import os
import random

import pytest

from openclatura import name_many
from openclatura.opsin_verify import verify_with_opsin
from openclatura.resonance_compare import canonical_smiles

pytestmark = [pytest.mark.dataset, pytest.mark.slow]

QM9_DATASET = "yairschiff/qm9"


@pytest.fixture(scope="module")
def qm9_indexed_sample():
    datasets = pytest.importorskip("datasets")
    n = int(os.environ.get("OPENCLATURA_DATASET_SAMPLE_N", "200"))
    seed = int(os.environ.get("OPENCLATURA_DATASET_SEED", "42"))
    assert n > 0, "OPENCLATURA_DATASET_SAMPLE_N must be positive"
    try:
        ds = datasets.load_dataset(QM9_DATASET, split="train")
    except Exception as exc:  # pragma: no cover - network-dependent
        pytest.skip(f"Could not load {QM9_DATASET}: {exc}")
    assert len(ds) > 0, f"{QM9_DATASET}: empty train split"
    indices = random.Random(seed).sample(range(len(ds)), min(n, len(ds)))
    return list(zip(indices, ds.select(indices)["smiles"], strict=True))


@pytest.fixture(scope="module")
def qm9_sample(qm9_indexed_sample):
    """Preserve the public fixture's sequence-of-SMILES interface."""
    return [smiles for _, smiles in qm9_indexed_sample]


def _assert_named(indexed_sample, results):
    assert indexed_sample, "QM9 gate requires a nonempty sample"
    assert len(results) == len(indexed_sample), (
        f"QM9 result count: expected={len(indexed_sample)} actual={len(results)}; indexed inputs={indexed_sample!r}"
    )
    failures = []
    for (index, smiles), result in zip(indexed_sample, results, strict=True):
        if not result:
            failures.append(f"index={index} smiles={smiles!r}: naming failure {result!r}")
    assert not failures, "QM9 naming failures:\n" + "\n".join(failures)


def _assert_exact_roundtrips(indexed_sample, results):
    _assert_named(indexed_sample, results)
    counts: dict[str, int] = {}
    failures = []
    unavailable = []
    for (index, smiles), result in zip(indexed_sample, results, strict=True):
        context = f"index={index} smiles={smiles!r} name={result.name!r}"
        try:
            # Use the original indexed input, never the naming result's SMILES.
            check = verify_with_opsin(result.name, smiles, standardize_smiles=False)
            status = check.status
            if status in {"skipped_no_opsin", "skipped_no_java"}:
                unavailable.append(f"{context}: {status}")
            elif status != "matched":
                failures.append(f"{context}: {status}; {check.error_message}")
            else:
                original = canonical_smiles(smiles)
                decoded = canonical_smiles(check.opsin_smiles)
                # The verifier's matched status also accepts some resonance forms.
                if original is None or decoded is None:
                    status = "exact_parseerror"
                elif original != decoded:
                    status = "exact_mismatched"
                else:
                    status = "exact_matched"
                if status != "exact_matched":
                    failures.append(f"{context}: {status}; original={original!r} roundtrip={decoded!r}")
        except Exception as exc:
            status = "error"
            failures.append(f"{context}: {type(exc).__name__}: {exc}")
        counts[status] = counts.get(status, 0) + 1

    assert not failures, "QM9 exact OPSIN failures:\n" + "\n".join(failures)
    if unavailable:
        pytest.skip("OPSIN / Java unavailable; exact gate incomplete:\n" + "\n".join(unavailable))
    return counts


def test_qm9_naming_rate(qm9_sample, qm9_indexed_sample, capsys):
    results = name_many(qm9_sample, processes=1)
    _assert_named(qm9_indexed_sample, results)
    named = sum(1 for r in results if r)
    total = len(results)
    with capsys.disabled():
        print(f"\n[qm9] sampled={total} named={named} rate={named / total:.2%}")


def test_qm9_opsin_match_rate(qm9_sample, qm9_indexed_sample, capsys):
    results = name_many(qm9_sample, processes=1)
    counts = _assert_exact_roundtrips(qm9_indexed_sample, results)

    with capsys.disabled():
        print(f"\n[qm9] opsin exact (canonical isomeric SMILES): {counts}; standardized: not evaluated")
