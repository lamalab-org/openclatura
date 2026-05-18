"""Round-trip tests for test_imines.py."""

import pytest

from structure_to_iupac.tests_roundtrip.roundtrip_helpers import roundtrip_smiles

SMILES = [
    'C=NC',
    'CC=NC',
    'CC(=N)C',
    'C1CCCCC1=N',
    'N1CCCCC1=N',
    'C1CCC2=CC=CC=C2C1=N',
]

@pytest.mark.parametrize("smiles", SMILES)
def test_roundtrip(smiles):
    roundtrip_smiles(smiles)

def test_has_smiles():
    if not SMILES:
        pytest.skip("No SMILES literals found.")
