"""Round-trip tests for test_substituents.py."""

import pytest

from openclatura.tests_roundtrip.roundtrip_helpers import roundtrip_smiles

SMILES = [
    "CC(C)CC",
    "CC(C)(C)CC",
    "CC(Cl)CO",
    "CS(C)(=O)=O",
    "CS(C)=O",
    "CSC",
    "C=CC1=CCCCC1",
]


@pytest.mark.parametrize("smiles", SMILES)
def test_roundtrip(smiles):
    roundtrip_smiles(smiles)


def test_has_smiles():
    if not SMILES:
        pytest.skip("No SMILES literals found.")
