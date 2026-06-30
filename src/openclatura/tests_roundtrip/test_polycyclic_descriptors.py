"""Round-trip tests for test_polycyclic_descriptors.py."""

import pytest

from openclatura.tests_roundtrip.roundtrip_helpers import roundtrip_smiles

SMILES = [
    "c1ccccc1",
    "O=S1CC2CCCCC2C1",
    "O=S1CCCC1",
    "C1=CC2CCCCC2C=C1",
    "C1=CC2CCC1C2",
]


@pytest.mark.parametrize("smiles", SMILES)
def test_roundtrip(smiles):
    roundtrip_smiles(smiles)


def test_has_smiles():
    if not SMILES:
        pytest.skip("No SMILES literals found.")
