"""Opt-in QM9 sampling test.

QM9 covers small organic molecules with up to 9 heavy atoms (C, N, O, F).
This test pulls a sample from a public QM9 mirror on Hugging Face
Datasets and reports naming + OPSIN match rates.

Marked ``dataset``/``slow``; run with ``pytest -m dataset``.
"""

from __future__ import annotations

import os
import random

import pytest

from bluenamer import name_many
from bluenamer.opsin_verify import verify_with_opsin

pytestmark = [pytest.mark.dataset, pytest.mark.slow]

# Common HF mirrors of QM9 with a SMILES column. The fixture tries them in
# order and skips if none are reachable. Override via env if needed.
_QM9_CANDIDATES = [
    ("jablonkagroup/qm9", "smiles"),
    ("yairschiff/qm9", "smiles"),
    ("mhsamavatian/qm9", "smiles"),
]


@pytest.fixture(scope="module")
def datasets_module():
    pytest.importorskip("datasets")
    import datasets

    return datasets


@pytest.fixture(scope="module")
def qm9_sample(datasets_module):
    n = int(os.environ.get("BLUENAMER_DATASET_SAMPLE_N", "200"))
    seed = int(os.environ.get("BLUENAMER_DATASET_SEED", "42"))
    override = os.environ.get("BLUENAMER_QM9_DATASET")
    candidates = [(override, "smiles")] if override else _QM9_CANDIDATES

    last_exc: Exception | None = None
    for repo, column in candidates:
        if repo is None:
            continue
        try:
            ds = datasets_module.load_dataset(repo, split="train")
            if column not in ds.column_names:
                continue
            random.seed(seed)
            sample = min(n, len(ds))
            indices = random.sample(range(len(ds)), sample)
            return ds.select(indices)[column]
        except Exception as exc:  # pragma: no cover - network-dependent
            last_exc = exc
            continue
    pytest.skip(f"No QM9 mirror reachable; last error: {last_exc}")


def test_qm9_naming_rate(qm9_sample, capsys):
    results = name_many(qm9_sample, processes=1)
    named = sum(1 for r in results if r.name)
    total = len(results)
    with capsys.disabled():
        print(f"\n[qm9] sampled={total} named={named} " f"name_rate={named / total:.2%}")
    # QM9 is dominated by simple structures; the floor is intentionally
    # higher than PubChem's.
    assert named / total > 0.50, f"QM9 naming rate dropped below 50%: {named}/{total}"


def test_qm9_opsin_match_rate(qm9_sample, capsys):
    results = name_many(qm9_sample, processes=1)
    matched = mismatched = unparseable = skipped = 0
    for r in results:
        if not r.name:
            continue
        check = verify_with_opsin(r.name, r.smiles)
        if check.status.startswith("skipped_"):
            skipped += 1
        elif check.status == "matched":
            matched += 1
        elif check.status == "mismatched":
            mismatched += 1
        elif check.status == "name_unparseable":
            unparseable += 1
    total = len(results)
    if skipped == sum(1 for r in results if r.name):
        pytest.skip("OPSIN / Java not available")
    with capsys.disabled():
        print(
            f"\n[qm9] opsin: matched={matched} mismatched={mismatched} "
            f"unparseable={unparseable} skipped={skipped}/{total}"
        )
