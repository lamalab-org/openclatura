"""Opt-in PubChem sample test.

Pulls a small random sample from ``jablonkagroup/pubchem-smiles-molecular-formula``
and reports naming + OPSIN match rates. Marked ``dataset``/``slow``;
run with ``pytest -m dataset``.

Tunable via env: ``BLUENAMER_DATASET_SAMPLE_N``, ``BLUENAMER_DATASET_SEED``.
"""

from __future__ import annotations

import os
import random

import pytest

from openclatura import name_many
from openclatura.opsin_verify import verify_with_opsin

pytestmark = [pytest.mark.dataset, pytest.mark.slow]

PUBCHEM_DATASET = "jablonkagroup/pubchem-smiles-molecular-formula"


@pytest.fixture(scope="module")
def pubchem_sample():
    datasets = pytest.importorskip("datasets")
    n = int(os.environ.get("BLUENAMER_DATASET_SAMPLE_N", "200"))
    seed = int(os.environ.get("BLUENAMER_DATASET_SEED", "42"))
    try:
        ds = datasets.load_dataset(PUBCHEM_DATASET, split="train")
    except Exception as exc:  # pragma: no cover - network-dependent
        pytest.skip(f"Could not load {PUBCHEM_DATASET}: {exc}")
    random.seed(seed)
    indices = random.sample(range(len(ds)), min(n, len(ds)))
    return ds.select(indices)["smiles"]


def test_pubchem_naming_rate(pubchem_sample, capsys):
    results = name_many(pubchem_sample, processes=1)
    named = sum(1 for r in results if r)
    total = len(results)
    with capsys.disabled():
        print(f"\n[pubchem] sampled={total} named={named} rate={named / total:.2%}")
    # PubChem is the headline coverage target; anything below 90% is a
    # regression that needs investigation, not just a "rate drift".
    assert named / total > 0.90, f"PubChem naming rate {named}/{total} = {named / total:.2%} below 90%"


def test_pubchem_opsin_match_rate(pubchem_sample, capsys):
    results = name_many(pubchem_sample, processes=1)
    counts: dict[str, int] = {}
    for r in results:
        if not r:
            continue
        status = verify_with_opsin(r.name, r.smiles).status
        counts[status] = counts.get(status, 0) + 1

    if counts.get("skipped_no_opsin") or counts.get("skipped_no_java"):
        pytest.skip("OPSIN / Java not available")

    with capsys.disabled():
        print(f"\n[pubchem] opsin: {counts}")
