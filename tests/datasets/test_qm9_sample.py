"""Opt-in QM9 sample test.

Pulls a small random sample from ``yairschiff/qm9`` and reports naming +
OPSIN match rates. Marked ``dataset``/``slow``; run with ``pytest -m dataset``.

Tunable via env: ``OPENCLATURA_DATASET_SAMPLE_N``, ``OPENCLATURA_DATASET_SEED``.
"""

from __future__ import annotations

import os
import random

import pytest

from openclatura import name_many
from openclatura.opsin_verify import verify_with_opsin

pytestmark = [pytest.mark.dataset, pytest.mark.slow]

QM9_DATASET = "yairschiff/qm9"


@pytest.fixture(scope="module")
def qm9_sample():
    datasets = pytest.importorskip("datasets")
    n = int(os.environ.get("OPENCLATURA_DATASET_SAMPLE_N", "200"))
    seed = int(os.environ.get("OPENCLATURA_DATASET_SEED", "42"))
    try:
        ds = datasets.load_dataset(QM9_DATASET, split="train")
    except Exception as exc:  # pragma: no cover - network-dependent
        pytest.skip(f"Could not load {QM9_DATASET}: {exc}")
    random.seed(seed)
    indices = random.sample(range(len(ds)), min(n, len(ds)))
    return ds.select(indices)["smiles"]


def test_qm9_naming_rate(qm9_sample, capsys):
    results = name_many(qm9_sample, processes=1)
    named = sum(1 for r in results if r)
    total = len(results)
    with capsys.disabled():
        print(f"\n[qm9] sampled={total} named={named} rate={named / total:.2%}")
    # QM9 is dominated by simple organics; floor flags total breakage.
    assert named / total > 0.50, f"QM9 naming rate dropped below 50%: {named}/{total}"


def test_qm9_opsin_match_rate(qm9_sample, capsys):
    results = name_many(qm9_sample, processes=1)
    counts: dict[str, int] = {}
    for r in results:
        if not r:
            continue
        status = verify_with_opsin(r.name, r.smiles).status
        counts[status] = counts.get(status, 0) + 1

    if counts.get("skipped_no_opsin") or counts.get("skipped_no_java"):
        pytest.skip("OPSIN / Java not available")

    with capsys.disabled():
        print(f"\n[qm9] opsin: {counts}")
