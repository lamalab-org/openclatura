"""Opt-in PubChem sampling test.

Pulls a small random sample from ``jablonkagroup/pubchem-smiles-molecular-formula``
on Hugging Face Datasets, runs the namer over it, and reports basic
metrics. Marked ``dataset`` (and ``slow``) so it is excluded from the
default pytest run.

Run with::

    pytest -m dataset

Requires the ``[datasets]`` extra and network access.
"""

from __future__ import annotations

import os
import random

import pytest

from bluenamer import name_many
from bluenamer.opsin_verify import verify_with_opsin

pytestmark = [pytest.mark.dataset, pytest.mark.slow]


@pytest.fixture(scope="module")
def datasets_module():
    pytest.importorskip("datasets")
    import datasets  # noqa: F401 — ensured importable

    return datasets


@pytest.fixture(scope="module")
def pubchem_sample(datasets_module):
    """Return ~200 SMILES sampled from the public PubChem mirror."""

    n = int(os.environ.get("BLUENAMER_DATASET_SAMPLE_N", "200"))
    seed = int(os.environ.get("BLUENAMER_DATASET_SEED", "42"))
    try:
        ds = datasets_module.load_dataset(
            "jablonkagroup/pubchem-smiles-molecular-formula",
            split="train",
        )
    except Exception as exc:  # pragma: no cover - network-dependent
        pytest.skip(f"Could not load PubChem dataset: {exc}")
    if len(ds) < n:
        n = len(ds)
    random.seed(seed)
    indices = random.sample(range(len(ds)), n)
    return ds.select(indices)["smiles"]


def test_pubchem_naming_rate(pubchem_sample, capsys):
    results = name_many(pubchem_sample, processes=1)
    named = sum(1 for r in results if r.name)
    errored = sum(1 for r in results if r.error)
    total = len(results)
    with capsys.disabled():
        print(f"\n[pubchem] sampled={total} named={named} errored={errored} " f"name_rate={named / total:.2%}")
    # Defensive minimum: at least 10% must yield a name. Real numbers
    # are much higher; the floor exists to flag total breakage.
    assert named / total > 0.10, f"PubChem naming rate dropped below 10%: {named}/{total}"


def test_pubchem_opsin_match_rate(pubchem_sample, capsys):
    results = name_many(pubchem_sample, processes=1)
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
            f"\n[pubchem] opsin: matched={matched} mismatched={mismatched} "
            f"unparseable={unparseable} skipped={skipped}/{total}"
        )
