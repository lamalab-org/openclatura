"""Round-trip tests for test_polycycles.py."""

import pytest

from structure_to_iupac.tests_roundtrip.roundtrip_helpers import roundtrip_smiles

SMILES = [
    "C1=CC2CCCCC2C1",
    "C1CCC2CCCCC2C1",
    "C1CC2CCC1C2",
    "CC1CC2CCC1C2",
    "N1CC2CCC1C2",
    "O1CC2CCC1C2",
    "C1CCC2(CC1)CCCCC2",
    "CC1CCC2(CC1)CCCCC2",
    "C1C2CC3CC1CC(C2)C3",
    "C12C3C4C1C1C2C3C41",
    "C1CN2CCC1CC2",
]


@pytest.mark.parametrize("smiles", SMILES)
def test_roundtrip(smiles):
    roundtrip_smiles(smiles)


def test_has_smiles():
    if not SMILES:
        pytest.skip("No SMILES literals found.")
