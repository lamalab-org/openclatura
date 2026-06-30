"""Round-trip tests for test_numbering.py."""

import pytest

from openclatura.tests_roundtrip.roundtrip_helpers import roundtrip_smiles

SMILES = [
    "CC(O)CC",
    "CCC(C)C",
]


@pytest.mark.parametrize("smiles", SMILES)
def test_roundtrip(smiles):
    roundtrip_smiles(smiles)


def test_has_smiles():
    if not SMILES:
        pytest.skip("No SMILES literals found.")
